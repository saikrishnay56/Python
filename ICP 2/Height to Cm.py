li=[]
N=int(input("N: No of students : "))
i=0
li1=[]
print("Enter heights in list:")
for i in range(N):
    li.append(float(input()))
    s=lambda f: f - f % 0.1
    li1.append(s(li[i] * 30.48))
print (li1)