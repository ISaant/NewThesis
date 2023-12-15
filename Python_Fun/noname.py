import numpy as np
import pandas as pd

x_train, x_test, y_train,y_test,_,_=Split(restStatePCA_s200,labels_s200,.3,seed=2)
x_train=Scale(x_train)
x_test=Scale(x_test)
model = Lasso(alpha=.2)
model.fit(x_train, y_train)
pred_Lasso=model.predict(x_test)
LassoPred=plotPredictionsReg(pred_Lasso,y_test,True)
print(LassoPred)

#%%



#%%

