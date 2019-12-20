import matplotlib.pyplot as plt

a=[]
b=[]


# y=0
# x=-50

for x in range(-50,50,1):
    y=x**2+2*x+2
    a.append(x)
    b.append(y)
    #x= x+1
#print(a)
#print(b)

#get poijnt where it intersects with the (y) we want
markers_on = []
for i in range(len(b)):
    y_val = b[i]
    x_val = a[i]
    #print(y_val)

    #if the x is the one we want, save thwe index
    if y_val == 1:
        #gets the x_val with the y_val we want
        print(x_val, y_val)
        #save the point id
        markers_on = [i]
fig= plt.figure()
axes=fig.add_subplot(111)
print(markers_on)
axes.plot(a,b,'-gD',markevery=markers_on)
plt.show()