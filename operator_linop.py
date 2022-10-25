import pyomo.environ as pyo
import pandas as pd
from io import StringIO

data = StringIO('''Operator;Wage;Mon;Tue;Wed;Thu;Fri;MinHours
K.C.; 25 ; 6 ; 0 ; 6 ; 0 ; 6 ; 8
D.H.; 26 ; 0 ; 6 ; 0 ; 6 ; 0 ; 8
H.B.; 24 ; 4 ; 8 ; 4 ; 0 ; 4 ; 8
S.C.; 23 ; 5 ; 5 ; 5 ; 0 ; 5 ; 8
K.S.; 28 ; 3 ; 0 ; 3 ; 8 ; 0 ; 7
N.K.; 30 ; 0 ; 0 ; 0 ; 6 ; 2 ; 7 ''')

df = pd.read_csv(data, sep=";", index_col='Operator')
days = ['Mon','Tue','Wed','Thu','Fri']
df[days] = df[days].astype(float)


def model_oxbridge():
  model = pyo.ConcreteModel()
  
  model.d = pyo.Set(initialize=days)
  model.i = pyo.Set(initialize= list(df.index))

  model.x = pyo.Var(model.i, model.d, within=pyo.NonNegativeIntegers)
  model.wage = pyo.Objective(expr=pyo.quicksum(df['Wage'][i]*(model.x[i,d]) for d in model.d for i in model.i), sense= pyo.minimize)

  def min_hours_rule(model,i):
    return pyo.quicksum(model.x[i,d] for d in model.d) >= df['MinHours'][i]
  model.min_hours = pyo.Constraint(model.i,rule= min_hours_rule)



  def day_total_rule(model,d):
    return pyo.quicksum(model.x[i,d] for i in model.i) ==14
  model.day_total = pyo.Constraint(model.d,rule=day_total_rule)

  def available_hours_rule(model,i,d):
    return model.x[i,d] <= df.loc[i,d]
  model.available_hours = pyo.Constraint(model.i,model.d, rule=available_hours_rule)

  return model
model_oxbridge()