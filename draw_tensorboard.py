from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val


def draw_plt(val, val_name, train, train_name):
    """将数据绘制成曲线图，val是数据，val_name是变量名称"""
    plt.figure()
    plt.plot([i.step for i in val], [j.value for j in val], label=val_name)
    plt.plot([i.step for i in train], [j.value for j in train], label=train_name)
    plt.legend()
    """横坐标是step，迭代次数
    纵坐标是变量值"""
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()


if __name__ == "__main__":
    tensorboard_path = 'lightning_logs/version_4/events.out.tfevents.1724809727.huqianjiangdeMacBook-Pro.local.62983.0'
    val = read_tensorboard_data(tensorboard_path, 'val_loss')
    train = read_tensorboard_data(tensorboard_path, 'train_loss')
    draw_plt(val, 'val_loss', train, 'train_loss', )