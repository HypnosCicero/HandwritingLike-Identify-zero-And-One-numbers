import torch.nn as nn
import xlrd

# Collecting data sets (搜集数据集)
def readAllData(filename):
    readxls = xlrd.open_workbook(filename)
    sheet1 = readxls.sheet_by_name('Data')
    nrows = sheet1.nrows
    for i in range(nrows):
        print(i)


filename = "E:\Work\Test_System\Identify_0_And_1_numbers\DataA.xlsx"
readAllData(filename)

# define the input number
N=0 # numbers of groups witch is the test data

model = nn.Sequential(
    nn.Linear(N,10,True),
    nn.ReLU(),
    nn.Linear(10,5,True),
    nn.ReLU(),
    nn.Linear(5,3,True),
    nn.ReLU(),
    nn.Linear(3,2)
)

