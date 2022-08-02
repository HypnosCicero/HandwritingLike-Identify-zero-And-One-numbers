import torch.nn
import torch.nn as nn
import xlrd

# Collecting data sets (搜集数据集)
def readAllData(filename):
    readxls = xlrd.open_workbook(filename)
    sheet1 = readxls.sheet_by_name('Data')
    cell = sheet1.cell_value(4, 10)
    print("cell = ")
    print(cell)
    nrows = sheet1.nrows
    '''
    for i in range(nrows):
        rowList=sheet1.row_values(i) # 按行遍历出来所有数据。
        print("这之前")
        print(rowList)
        for j in range(rowList.__sizeof__()):
            #k = 0
            print("list内部")
            print(rowList[j])
        #print('在这之后:' + rowList)
    '''


filename = "E:\Work\Test_System\Identify_0_And_1_numbers\DataA.xlsx"
readAllData(filename)

# define x and y
x = 100
y = 100


# define the input number
N = 10  # numbers of groups witch is the test data

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

