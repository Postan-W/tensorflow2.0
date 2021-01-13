
def search_sequence(array,k):
    #计数，也就是时间复杂度
    count = 0
    #定义目标序列的左端与右端
    i =0
    j =0
    while j <len(array):
        sum = 0
        for t in range(i,j+1):
            sum += array[t]
            count += 1
        if sum == k:
            print("已找到该序列,左端是%d，右端是%d"%(i,j))
            break
        elif sum < k:
            j += 1
            continue
        else:
            i += 1
            if i > j:
                j = i
    if j >= len(array):
        print("不存在")

#测试
a = [1,2,3,4,5,6,7,8,9,10]
#设置k为22，这个解是唯一的：i=4,j=6
search_sequence(a,1000)