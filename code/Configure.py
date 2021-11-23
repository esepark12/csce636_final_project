# Below configures are examples,
# you can modify them as you wish.

### YOUR CODE HERE
import ImageUtils

training_configs = {
	"experiment_name": 'test-ResNetProp-ImageStandard',
	"train_augmentation" : ImageUtils.parse_record(None,training=1),
	"batch_size" : 128,
	"initial_lr": 0.1,
	"learn_rate_schedule" : {80: 0.1, 110: 0.05, 200: 0.002, 300: 0.001, 350: 0.0001},
	"max_epoch": 350,
}
model_configs = {
	"name": 'MyModel',
	"depth": 2,
	"save_dir": '../saved_models/',
}

### END CODE HERE
