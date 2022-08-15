import torch.nn
import torch.nn as nn
import xlrd

# Collecting data sets (搜集数据集)

X = torch.tensor([64, 12])
Y = torch.tensor([12])
def readAllData(filename):
    readXls = xlrd.open_workbook(filename)
    sheet1 = readXls.sheet_by_name('Data')
    data_row = 4
    data_column = 10
    value_row = 8
    value_column = 13
    # cell = sheet1.cell_value(4, 10)
    XList = []
    YList = []
    for i in range(64):
        xlist = []
        if i > 0 :
            data_row -= 4
            data_column += 4
        myFlag = 0
        for j in range(12):
            xvalue = sheet1.cell_value(data_row, data_column)
            print("x value is " + str(xvalue))
            xlist.append(xvalue)
            myFlag += 1
            if myFlag == 3:
                data_column -= 2
                data_row += 1
                myFlag = 0
            else:
                data_column += 1
        XList.append(xlist)
        yvalue = sheet1.cell_value(value_row, value_column)
        YList.append(yvalue)
        print("y value is " + str(yvalue))
        value_column += 4
        print("\n")

    X = torch.tensor(XList, dtype=torch.long)
    Y = torch.tensor(YList, dtype=torch.long)
    print(X)
    print(Y)


filename = "E:\Work\Test_System\Identify_0_And_1_numbers\DataA.xlsx"
readAllData(filename)


# define x and y
x = 100
y = 100



# define the input number
N = 10  # numbers of groups witch is the test data

# in this model the N is  3*4(meening is 3c 4r)

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
    y_pred = model(x)

    # The predicted values of y and y are calculated using the loss function to work out the deviation values (利用损失函数对y与y的预测值进行计算，并得出偏差值)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Using optimizer to clear grad of modle (利用optimizer进行对偏导清零操作)
    optimizer.zero_grad()

    # Reverse Propagation (反向传播)
    loss.backward()

    # Optimization parameters (优化参数)
    optimizer.step()

