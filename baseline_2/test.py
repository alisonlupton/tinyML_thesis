import torch
import torchvision
from avalanche.benchmarks.datasets import MNIST
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset
import inspect

# Most datasets in Avalanche are automatically downloaded the first time you use them
# and stored in a default location. You can change this folder by calling
# avalanche.benchmarks.utils.set_dataset_root(new_location)
datadir = default_dataset_location('mnist')

# As we would simply do with any Pytorch dataset we can create the train and 
# test sets from it. We could use any of the above imported Datasets, but let's
# just try to use the standard MNIST.
train_MNIST = MNIST(datadir, train=True, download=True)
test_MNIST = MNIST(datadir, train=False, download=True)

# transformations are managed by the AvalancheDataset
train_transforms = torchvision.transforms.ToTensor()
eval_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((32, 32))
])

# wrap datasets into Avalanche datasets
# notice that AvalancheDatasets have multiple transform groups
# `train` and `eval` are the default ones, but you can add more (e.g. replay-specific transforms)
train_MNIST = as_classification_dataset(
    train_MNIST,
    transform_groups={
        'train': train_transforms, 
        'eval': eval_transforms
    }
)
test_MNIST = as_classification_dataset(
    test_MNIST,
    transform_groups={
        'train': train_transforms, 
        'eval': eval_transforms
    }
)

# we can iterate the examples as we would do with any Pytorch dataset
for i, example in enumerate(train_MNIST):
    print(f"Sample {i}: {example[0].shape} - {example[1]}")
    break

# or use a Pytorch DataLoader
train_loader = torch.utils.data.DataLoader(
    train_MNIST, batch_size=32, shuffle=False
)
for i, (x, y) in enumerate(train_loader):
    print(f"Batch {i}: {x.shape} - {y.shape}")
    break

# we can also switch between train/eval transforms
train_MNIST.train()
print(train_MNIST[0][0].shape)

train_MNIST.eval()
print(train_MNIST[0][0].shape)

print(len(train_MNIST))  # 60k
print(len(train_MNIST.concat(train_MNIST)))  # 120k





desired_classes = [0, 1, 2, 3, 4]
# train_MNIST.targets is a list of ints (one per sample)
idx_train_0_4 = [i for i, y in enumerate(train_MNIST.targets)
                 if y in desired_classes]

idx_test_0_4  = [i for i, y in enumerate(test_MNIST.targets)
                 if y in desired_classes]



# subsampling is often used to create streams or replay buffers!
dsub_train = train_MNIST.subset(idx_train_0_4)
dsub_test  = test_MNIST.subset(idx_test_0_4)

print(len(dsub_train))  # 5
# targets are preserved when subsetting
print(list(dsub_train.targets))
print(list(dsub_test.targets))







from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark

bm = class_incremental_benchmark({'train': dsub_train, 'test': dsub_test}, num_experiences=5)

exp = bm.train_stream[0]
print(f"Experience {exp.current_experience}")
print(f"Classes in this experience: {exp.classes_in_this_experience}")
print(f"Previous classes: {exp.classes_seen_so_far}")
print(f"Future classes: {exp.future_classes}")

print(f"{bm.train_stream.name} - len {len(bm.train_stream)}")
print(f"{bm.test_stream.name} - len {len(bm.test_stream)}")

for exp in bm.train_stream:
     print(exp.current_experience, exp.classes_in_this_experience)

