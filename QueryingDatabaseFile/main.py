import pandas as pd
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

#os.environ["OPENAI_API_KEY"] = getpass.getpass()

#df = pd.read_excel("crime.xlsx")
#print(df.shape)
#print(df.columns.tolist())

engine = create_engine("sqlite:///crime.db")
#df.to_sql("crime", engine, index=False)

db = SQLDatabase(engine=engine)
print(db.dialect)
print(db.get_usable_table_names())

db.run("SELECT * FROM crime WHERE Year < 2021;")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)