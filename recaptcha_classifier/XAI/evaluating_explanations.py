import os
import torch
import quantus
import numpy as np

dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
x_batch, y_batch = next(iter(dataloader))
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()


print(a_batch_saliency_ce_model.shape)
#print(type(a_batch_saliency_ce_model))


model = load_model('model_cross_entropy')
model.eval()

metric = quantus.IROF(return_aggregate=False)
print(metric.get_params)

ce_scores = metric(
    model=model,
    channel_first=True,
    x_batch=x_batch,
    y_batch=y_batch,
    a_batch=a_batch_test,
    device= check_device(),
    explain_func=quantus.explain,
    explain_func_kwargs={"method": "Saliency"}
)
print(len(ce_scores))
np.savetxt("ce_scores.csv", ce_scores, delimiter=",")