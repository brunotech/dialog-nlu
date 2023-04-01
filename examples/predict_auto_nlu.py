# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""


from dialognlu import AutoNLU


model_path = "../saved_models/joint_distilbert_model"

print("Loading model ...")
nlu = AutoNLU.load(model_path)

print("Prediction ...")
utterance = "add sabrina salerno to the grime instrumentals playlist"
print(f"utterance: {utterance}")
result = nlu.predict(utterance)
print(f"result: {result}")