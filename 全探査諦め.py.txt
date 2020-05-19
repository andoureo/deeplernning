field = [[0 for i in range(4)] for j in range(3)]
field[2][1] = "A"
field[2][2] = "B"
field[0][1] = "C"
ancer = [[0,0,0,0],[0,0,0,0],[3,"A","C","B"]]

def count(field):
    for i in range(3):
        count = 0
        for j in range(1,4):
            if field[i][j] != 0:
                count +=1
            if j == 3:
                field[i][0] = count
    return field

def move(field):
    first=[]
    path=[]
    while (field != ancer):
        num = 0
        for i in range(3):
            for j in range(1,4):
                if not field[i][j] == ancer[i][j]:
                    temp_field=field
                    for k in range(3):
                        if field[k][0]<3:
                            temp_field[((i+1)%3)][((field[k][0])+1)] = field[i][j]
                            temp_field[i][j] = 0
                            print(path)
                            print(temp_field)
                            if not temp_field in path and not temp_field in first:
                                print(field)
                                field=count(temp_field)
                                path.append(field)
                                num +=1
                                if(num<2):
                                    first.append(field)
                                else :
                                    temp_field=field
    return num
    
field = count(field)
temp_anc = move(field)
print(temp_anc)
