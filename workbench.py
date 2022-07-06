# %% 
from sqlalchemy import create_engine, inspect

# %%
DATABASE_SCHEMA='mlpdb'
DATABASE_USER='mlp'
DATABASE_PASS='mlp'
DATABASE_HOST='artemis.idsia.ch:15432'

engine = create_engine(f'postgresql://{DATABASE_USER}:{DATABASE_PASS}@{DATABASE_HOST}/{DATABASE_SCHEMA}')

# %%
with engine.connect() as conn:
    rs = conn.execute('SELECT * FROM predictions')

    for row in rs:
        print(row)

# %%
inspector = inspect(engine)
inspector.get_columns('predictions')
# %%
