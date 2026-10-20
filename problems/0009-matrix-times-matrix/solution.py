def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:


                if len(a[0]) != len(b):
                    return -1

                c=[]

                for i in range(len(a)):
                    row = []
                    for j in range(len(a[i])):
                        sum = 0
                        for k in range(len(b)):
                            sum += a[i][k] * b[k][j]
                        row.append(sum)
                    c.append(row)
                        
                        
                    

                return c