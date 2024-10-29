from datasets import load_dataset

dataset = load_dataset("Lin-Chen/MMStar", "val")

# take a close look of evaluation samples
print(dataset["val"][0])
dataset["val"][0]['image']  # display the image