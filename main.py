import torch.nn
import torch.nn as nn
import xlrd
# for train model  64 is all of number  52 + 14 it has two numbers (0 and 1) is Repetitive
# 因为为了训练模型所以64个数被分为了52和14 两组（满足28定理）又因为有一个数字作为重复的即做训练也做测试

# 26 and 7 are represented as 26 data in the number '0' as training and 7 data as test
# 26和7表示为数字‘0’中26个数据作为训练，7个数据作为测试
trainX = torch.tensor([52, 12])
trainY = torch.tensor([52])

testX = torch.tensor([14, 12])
testY = torch.tensor([14])
# Collecting data sets And Processing the data (搜集数据集，并将其处理)
def readAndProcessingAllData(filename):
    # Use the list and add the data to the trainTensor to train the neural network 利用list，并将数据加入到trainTensor中从而进行神经网络的训练
    trainXList = []
    trainYList = []

    # Use the list and add the data to the testTensor to test the neural network 利用list，并将数据加入到testTensor中从而进行神经网络的测试
    testXList = []
    testYList = []

    readXls = xlrd.open_workbook(filename)
    sheet1 = readXls.sheet_by_name('Data')
    data_row = 4
    data_column = 10
    value_row = 8
    value_column = 13
    deviceTrandTeF = 0

    for i in range(64):
        xlist = []
        if i > 0:
            data_row -= 4
            data_column += 4
        colleController = 0 # this controller is using the data colleciton 数据搜集控制器
        for j in range(12):
            xvalue = sheet1.cell_value(data_row, data_column)
            xlist.append(xvalue)
            colleController += 1
            if colleController == 3:
                data_column -= 2
                data_row += 1
                colleController = 0
            else:
                data_column += 1

        yvalue = sheet1.cell_value(value_row, value_column)

        # 25 is 26(25就是1开始的26)
        if deviceTrandTeF%32 <= 25:
            trainXList.append(xlist)
            trainYList.append(yvalue)
        if deviceTrandTeF%32 >=25:
            testXList.append(xlist)
            testYList.append(yvalue)

        value_column += 4

        deviceTrandTeF += 1

    trainX = torch.tensor(trainXList, dtype=torch.long)
    trainY = torch.tensor(trainYList, dtype=torch.long)

    print(trainX)
    print(trainY)

    testX = torch.tensor(testXList, dtype=torch.long)
    testY = torch.tensor(testYList, dtype=torch.long)

    print(testX)
    print(testY)


filename = "E:\Work\Test_System\Identify_0_And_1_numbers\DataA.xlsx"
readAndProcessingAllData(filename)



# define the input number
N = 12  # numbers of groups witch is the test data

# in this model the N is  4*3(meening is 4r 3c)

model = nn.Sequential(
    nn.Linear(N, 10, True),
    nn.ReLU(),
    nn.Linear(10, 5, True),
    nn.ReLU(),
    nn.Linear(5, 3, True),
    nn.ReLU(),
    nn.Linear(3, 2)
)

# define loss function (定义损失函数)
loss_fn = torch.nn.MSELoss(reduction='sum') # 均方差函数

# define learning rate (定义学习率)
learning_rate=1e-4

# make an object who is optimizer (构造一个optimizer对象)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# The process of training (训练的过程)
for t in range(500):
    # Passing the values into the neural network the predicted value of y is calculated (将数值传入神经网络中，并将y的预测值算出)
    y_pred = model(trainX)  # have some bugs

    # The predicted values of y and y are calculated using the loss function to work out the deviation values (利用损失函数对y与y的预测值进行计算，并得出偏差值)
    loss = loss_fn(y_pred, trainY)
    print(t, loss.item())

    # Using optimizer to clear grad of modle (利用optimizer进行对偏导清零操作)
    optimizer.zero_grad()

    # Reverse Propagation (反向传播)
    loss.backward()

    # Optimization parameters (优化参数)
    optimizer.step()


# test Modle
torch.no_grad()
outputs = model(testX)
print("this is the testX:")
print(testX)
print("\n")
print("test modle:")
print(outputs)

print("testY:")
print(testY)