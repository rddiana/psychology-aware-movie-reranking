"""KION reward regression. All historical features use dates strictly before cutoff."""
from __future__ import annotations
import json, hashlib, math
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

VERSION = 'kion-reward-1.0'
CAT_FEATURES = ['content_type', 'main_genre', 'main_country']
NUM_FEATURES = ['release_year', 'age_rating', 'genre_count',
    'u_log_count', 'i_log_count', 'g_log_count', 'ug_log_count',
    'u_reward', 'i_reward', 'g_reward', 'ug_reward', 'global_reward',
    'u_std', 'u_completion', 'u_zero', 'i_completion', 'i_zero',
    'u_recency', 'i_recency', 'ug_affinity', 'u_missing', 'i_missing', 'metadata_missing']
FEATURES = CAT_FEATURES + NUM_FEATURES

def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def sql_path(path): return "'"+str(Path(path).resolve()).replace('\\','/').replace("'","''")+"'"

def sanitize_features(df):
    out=df[FEATURES].copy()
    for c in CAT_FEATURES: out[c]=out[c].fillna('__unknown__').astype(str)
    for c in NUM_FEATURES:
        out[c]=pd.to_numeric(out[c],errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
    return out

class FeatureStore:
    def __init__(self, db_path, read_only=False, memory_limit='2GB', threads=4):
        self.db_path=Path(db_path)
        self.con=duckdb.connect(str(db_path),read_only=read_only)
        self.con.execute(f"SET memory_limit='{memory_limit}'")
        self.con.execute(f"SET threads={int(threads)}")
        self.con.execute('SET preserve_insertion_order=false')

    def close(self): self.con.close()

    def ingest(self, data_dir, metadata_path, mapping, splits=('train', 'validation', 'test')):
        """Load only explicitly requested temporal splits."""
        data_dir=Path(data_dir)
        selects=[]
        for split in splits:
            path=data_dir/f'{split}_interactions.csv'
            selects.append(f"""SELECT CAST(user_id AS BIGINT) user_id, CAST(item_id AS BIGINT) item_id,
                CAST(last_watch_dt AS DATE) event_day, TRY_CAST(watched_pct AS DOUBLE) watched_pct,
                CAST(source_row_index AS BIGINT) row_id, '{split}' split
                FROM read_csv_auto({sql_path(path)}, sample_size=-1)""")
        self.con.execute('CREATE OR REPLACE TABLE events AS '+' UNION ALL '.join(selects))
        assert self.con.execute('SELECT count(*)=count(DISTINCT row_id) FROM events').fetchone()[0], 'row_id must be unique'
        assert self.con.execute('SELECT count(*) FROM events WHERE user_id IS NULL OR item_id IS NULL OR event_day IS NULL').fetchone()[0]==0
        self.con.execute('ALTER TABLE events ADD COLUMN y DOUBLE')
        self.con.execute('UPDATE events SET y=CASE WHEN isfinite(watched_pct) AND watched_pct BETWEEN 0 AND 100 THEN watched_pct/100.0 ELSE NULL END')
        meta=pd.read_csv(metadata_path)
        assert not meta.item_id.duplicated().any()
        for c in CAT_FEATURES: meta[c]=meta[c].fillna('__unknown__').astype(str)
        self.con.register('_meta',meta)
        self.con.execute('CREATE OR REPLACE TABLE metadata AS SELECT * FROM _meta')
        self.con.unregister('_meta')
        vocab=pd.DataFrame({'item_id':list(map(int,mapping['item_id_to_index']))})
        self.con.register('_vocab',vocab)
        self.con.execute('CREATE OR REPLACE TABLE policy_vocab AS SELECT * FROM _vocab')
        self.con.unregister('_vocab')

    def audit(self, boundaries):
        report=self.con.execute("""SELECT split, count(*) n_rows, count(y) valid_labels,
            count(*) FILTER(WHERE watched_pct IS NULL) missing,
            count(*) FILTER(WHERE watched_pct IS NOT NULL AND NOT isfinite(watched_pct)) nonfinite,
            count(*) FILTER(WHERE watched_pct<0) negative,
            count(*) FILTER(WHERE watched_pct>100) above100,
            count(*) FILTER(WHERE watched_pct=0) zeros,
            count(*) FILTER(WHERE watched_pct=100) hundreds,
            min(watched_pct) min_pct, max(watched_pct) max_pct,
            avg(watched_pct) mean_pct, median(watched_pct) median_pct,
            min(event_day) start_day, max(event_day) finish_day
            FROM events GROUP BY split ORDER BY start_day""").df()
        for split, b in boundaries.items():
            invalid=self.con.execute('SELECT count(*) FROM events WHERE split=? AND (event_day<? OR event_day>?)',
                                     [split,b['start'],b['end']]).fetchone()[0]
            assert invalid==0, f'{split}: даты не соответствуют GPT2 split'
        dup=self.con.execute('SELECT count(*)-count(DISTINCT (user_id,item_id)) FROM events').fetchone()[0]
        return report, int(dup)

    def build_histories(self, end_exclusive):
        # Final feature store intentionally contains train+validation only, never test labels.
        self.con.execute("""CREATE OR REPLACE TABLE history_source AS
            SELECT e.*, coalesce(m.main_genre,'__unknown__') main_genre
            FROM events e LEFT JOIN metadata m USING(item_id) WHERE event_day < CAST(? AS DATE)""",[end_exclusive])
        self.con.execute('CREATE OR REPLACE TABLE state_info AS SELECT CAST(? AS DATE) history_end_exclusive', [end_exclusive])
        for name, keys in [('global',[]),('user',['user_id']),('item',['item_id']),
                            ('genre',['main_genre']),('usergenre',['user_id','main_genre'])]:
            key_sql=','.join(keys)
            group_sql=key_sql+',' if keys else ''
            window=('PARTITION BY '+key_sql+' ' if keys else '')+'ORDER BY hist_day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW'
            self.con.execute(f"""CREATE OR REPLACE TABLE h_{name} AS
              WITH daily AS (SELECT {group_sql} event_day hist_day, count(*) n, count(y) n_y,
                coalesce(sum(y),0) s, coalesce(sum(y*y),0) s2,
                count(*) FILTER(WHERE y>=0.8) complete, count(*) FILTER(WHERE y=0) zero
                FROM history_source GROUP BY {group_sql} event_day)
              SELECT {group_sql} hist_day,
                sum(n) OVER w n, sum(n_y) OVER w n_y, sum(s) OVER w s,
                sum(s2) OVER w s2, sum(complete) OVER w complete, sum(zero) OVER w zero
                FROM daily WINDOW w AS ({window})""")
        self.con.execute('DROP TABLE history_source')
        self.con.execute('CHECKPOINT')

    def candidate_query(self, split, boundaries, max_rows=None):
        assert split in {'train','validation','test'}
        cutoff='event_day' if split=='train' else f"DATE '{boundaries[split]['start']}'"
        sample='' if max_rows is None else f'ORDER BY hash(row_id),row_id LIMIT {int(max_rows)}'
        return f"""SELECT row_id, user_id, item_id, event_day, {cutoff} cutoff, y
             FROM events WHERE split='{split}' AND y IS NOT NULL {sample}"""

    def feature_sql(self, query):
        return f"""WITH c0 AS ({query}), c AS (
            SELECT c0.*, coalesce(m.content_type,'__unknown__') content_type,
              coalesce(m.main_genre,'__unknown__') main_genre,
              coalesce(m.main_country,'__unknown__') main_country,
              m.release_year, m.age_rating, m.genre_count,
              CASE WHEN m.item_id IS NULL THEN 1 ELSE 0 END metadata_missing
            FROM c0 LEFT JOIN metadata m USING(item_id)),
          j AS (SELECT c.*, g.s gs,g.n_y gn,
              u.n un,u.n_y uyn,u.s us,u.s2 us2,u.complete uc,u.zero uz,u.hist_day ud,
              i.n itn,i.n_y iyn,i.s its,i.complete ic,i.zero iz,i.hist_day iday,
              ge.n gen,ge.n_y geyn,ge.s ges,
              ug.n ugn,ug.n_y ugyn,ug.s ugs
            FROM c ASOF LEFT JOIN h_global g ON c.cutoff>g.hist_day
              ASOF LEFT JOIN h_user u ON c.user_id=u.user_id AND c.cutoff>u.hist_day
              ASOF LEFT JOIN h_item i ON c.item_id=i.item_id AND c.cutoff>i.hist_day
              ASOF LEFT JOIN h_genre ge ON c.main_genre=ge.main_genre AND c.cutoff>ge.hist_day
              ASOF LEFT JOIN h_usergenre ug ON c.user_id=ug.user_id AND c.main_genre=ug.main_genre AND c.cutoff>ug.hist_day),
          p AS (SELECT *,coalesce(gs/nullif(gn,0),0.5) history_prior FROM j)
          SELECT row_id,user_id,item_id,event_day,cutoff,y,content_type,main_genre,main_country,
            release_year,age_rating,genre_count,metadata_missing,
            ln(1+coalesce(un,0)) u_log_count, ln(1+coalesce(itn,0)) i_log_count,
            ln(1+coalesce(gen,0)) g_log_count, ln(1+coalesce(ugn,0)) ug_log_count,
            (coalesce(us,0)+5*history_prior)/(coalesce(uyn,0)+5) u_reward,
            (coalesce(its,0)+20*history_prior)/(coalesce(iyn,0)+20) i_reward,
            (coalesce(ges,0)+20*history_prior)/(coalesce(geyn,0)+20) g_reward,
            (coalesce(ugs,0)+3*history_prior)/(coalesce(ugyn,0)+3) ug_reward,
            history_prior global_reward,
            sqrt(greatest(0,coalesce(us2/nullif(uyn,0)-pow(us/nullif(uyn,0),2),0))) u_std,
            (coalesce(uc,0)+5*history_prior)/(coalesce(uyn,0)+5) u_completion,
            coalesce(uz/nullif(uyn,0),0) u_zero,
            (coalesce(ic,0)+20*history_prior)/(coalesce(iyn,0)+20) i_completion,
            coalesce(iz/nullif(iyn,0),0) i_zero,
            coalesce(date_diff('day',ud,cutoff),9999) u_recency,
            coalesce(date_diff('day',iday,cutoff),9999) i_recency,
            coalesce(ugn/nullif(un,0),0) ug_affinity,
            CASE WHEN un IS NULL THEN 1 ELSE 0 END u_missing,
            CASE WHEN itn IS NULL THEN 1 ELSE 0 END i_missing,
            item_id IN (SELECT item_id FROM policy_vocab) in_policy_vocab
          FROM p"""

    def export_features(self,split,boundaries,path,max_rows=None):
        q=self.feature_sql(self.candidate_query(split,boundaries,max_rows))
        self.con.execute(f'COPY ({q} ORDER BY event_day,row_id) TO {sql_path(path)} (FORMAT PARQUET, COMPRESSION ZSTD)')

    def features_for_candidates(self,user_id,item_ids,cutoff):
        end=self.con.execute('SELECT history_end_exclusive FROM state_info').fetchone()[0]
        cutoff=pd.Timestamp(cutoff).date()
        if cutoff>end:
            raise ValueError('FeatureStore не содержит историю до этого cutoff; обновите store по реальным логам.')
        candidates=pd.DataFrame({'row_id':np.arange(len(item_ids),dtype='int64'),
            'user_id':int(user_id),'item_id':np.asarray(item_ids,dtype='int64'),
            'event_day':pd.Timestamp(cutoff),'cutoff':pd.Timestamp(cutoff),'y':np.nan})
        self.con.register('_candidates',candidates)
        try:return self.con.execute(self.feature_sql('SELECT * FROM _candidates')+' ORDER BY row_id').df()
        finally:self.con.unregister('_candidates')

    def remove_outcomes(self):
        # Serving artifact needs past feature tables and metadata, not held-out outcomes.
        self.con.execute('DROP TABLE events')
        self.con.execute('CHECKPOINT')

def regression_metrics(y,pred,users):
    y=np.asarray(y,dtype=float); raw=np.asarray(pred,dtype=float)
    assert np.isfinite(y).all() and np.isfinite(raw).all()
    clipped=np.clip(raw,0,1)
    err=clipped-y
    d=pd.DataFrame({'user':users,'ae':abs(err),'se':err**2})
    macro=d.groupby('user',sort=False)[['ae','se']].mean()
    return {'MAE':float(abs(err).mean()),'RMSE':float(np.sqrt(np.mean(err**2))),
        'MAE_percentage_points':float(100*abs(err).mean()),
        'RMSE_percentage_points':float(100*np.sqrt(np.mean(err**2))),
        'macro_user_MAE':float(macro.ae.mean()),
        'raw_MAE':float(np.mean(abs(raw-y))), 'raw_RMSE':float(np.sqrt(np.mean((raw-y)**2))),
        'prediction_outside_01':float(np.mean((raw<0)|(raw>1)))}

def ranking_metrics(frame, predictions, ks=(5,10), pair_cap=256, seed=42):
    """Same group metrics as the reference, with vectorized ranking and bounded pair memory."""
    df=frame[['row_id','user_id','item_id','cutoff','y']].copy()
    names=list(predictions)
    for name,p in predictions.items():df[name]=np.clip(np.asarray(p),0,1)
    if df.groupby('user_id').cutoff.nunique().max()>1:raise ValueError('Ranking requires one cutoff per user')
    source_users=df.user_id.nunique()
    df=df.sort_values('user_id',kind='stable')
    users,starts,counts=np.unique(df.user_id.to_numpy(),return_index=True,return_counts=True)
    y=df.y.to_numpy(dtype=float)
    eligible=(counts>=2)&(np.maximum.reduceat(y,starts)>np.minimum.reduceat(y,starts))
    df=df.iloc[np.flatnonzero(np.repeat(eligible,counts))]
    if df.empty:return pd.DataFrame(),pd.DataFrame()
    users,starts,counts=np.unique(df.user_id.to_numpy(),return_index=True,return_counts=True)
    n_groups=len(users); group=np.repeat(np.arange(n_groups),counts)
    y=df.y.to_numpy(dtype=float);items=df.item_id.to_numpy();row_ids=df.row_id.to_numpy()
    position=np.arange(len(df))-np.repeat(starts,counts)
    discounts=1/np.log2(position+2)
    ideal=np.lexsort((row_ids,items,-y,group))
    idcg={k:np.bincount(group,weights=y[ideal]*discounts*(position<k),minlength=n_groups) for k in ks}
    group_mean=np.bincount(group,weights=y,minlength=n_groups)/counts
    oracle=np.maximum.reduceat(y,starts)
    pa=[];pb=[];rng=np.random.default_rng(seed)
    for start,n in zip(starts,counts):
        if n*(n-1)//2<=pair_cap:a,b=np.triu_indices(n,1)
        else:
            a=rng.integers(0,n,size=pair_cap*3);b=rng.integers(0,n,size=pair_cap*3)
            take=a!=b;a,b=a[take][:pair_cap],b[take][:pair_cap]
        take=y[start+a]!=y[start+b]
        pa.append(start+a[take]);pb.append(start+b[take])
    a=np.concatenate(pa);b=np.concatenate(pb);pg=group[a]
    pair_counts=np.bincount(pg,minlength=n_groups)
    records=[]
    for name in names+['random_expected']:
        block={'user_id':users,'model':name,'candidates':counts,'pairs':pair_counts,
               'OracleTop1Reward':oracle}
        if name=='random_expected':
            block['PairAccuracy']=np.where(pair_counts>0,.5,np.nan)
            block['Top1Reward']=group_mean
        else:
            s=df[name].to_numpy(dtype=float)
            order=np.lexsort((row_ids,items,-s,group))
            pair_score=np.where(s[a]==s[b],.5,((s[a]-s[b])*(y[a]-y[b])>0).astype(float))
            pair_sum=np.bincount(pg,weights=pair_score,minlength=n_groups)
            block['PairAccuracy']=np.divide(pair_sum,pair_counts,out=np.full(n_groups,np.nan),where=pair_counts>0)
            block['Top1Reward']=y[order[starts]]
        for k in ks:
            if name=='random_expected':
                dcg=group_mean*np.bincount(group,weights=discounts*(position<k),minlength=n_groups)
            else:
                dcg=np.bincount(group,weights=y[order]*discounts*(position<k),minlength=n_groups)
            block[f'NDCG@{k}']=dcg/idcg[k]
        records.append(pd.DataFrame(block))
    detail=pd.concat(records,ignore_index=True)
    cols=['PairAccuracy','Top1Reward','OracleTop1Reward']+[f'NDCG@{k}' for k in ks]
    result=detail.groupby('model')[cols].mean()
    result['users']=n_groups;result['source_users']=source_users
    result['coverage_users']=n_groups/source_users
    return result.reset_index(),detail

def calibration_table(y,pred):
    f=pd.DataFrame({'y':y,'prediction':np.clip(pred,0,1)})
    f['bin']=pd.cut(f.prediction,np.linspace(0,1,11),include_lowest=True)
    return f.groupby('bin',observed=False).agg(n=('y','size'),predicted=('prediction','mean'),observed=('y','mean')).reset_index()

class RewardModel:
    def __init__(self,artifact_dir):
        from catboost import CatBoostRegressor
        self.root=Path(artifact_dir)
        self.contract=json.loads((self.root/'reward_contract.json').read_text(encoding='utf-8'))
        assert self.contract['version']==VERSION
        self.model=CatBoostRegressor(); self.model.load_model(str(self.root/'reward_model.cbm'))
        self.store=FeatureStore(self.root/'reward_features.duckdb',read_only=True)

    def predict(self,user_id,item_ids,cutoff):
        f=self.store.features_for_candidates(user_id,item_ids,cutoff)
        p=np.clip(self.model.predict(sanitize_features(f)),0,1)
        return pd.DataFrame({'item_id':f.item_id.to_numpy(),'reward':p,
                             'cold_user':f.u_missing.to_numpy().astype(bool),
                             'cold_item':f.i_missing.to_numpy().astype(bool),
                             'in_policy_vocab':f.in_policy_vocab.to_numpy().astype(bool)})

    def close(self):self.store.close()
