from torch.utils.data.dataloader import default_collate

"""
    dirty_collate

    Dirty fix for NoneTypes in the batch. This is a dirty fix, but it works.

    https://discuss.pytorch.org/t/questions-about-dataloader-and-dataset/806/4
"""
def dirty_collate(batch):
    batch = list(filter (lambda x:(x[0] is not None and x[1] is not None), batch))
    return default_collate(batch)