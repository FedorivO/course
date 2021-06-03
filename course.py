import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import matplotlib.patches as mpatches
counter1 = 0


global_points = []

def evaluate_f(x1, x2, R):
    global counter1
    counter1 += 1
    if (x1)**2 + x2**2 <= 1  and (x1-0.4)**2 + x2**2 >= 0.09:
        return (100*(x1**2 - x2)**2) + (x1 - 1)**2
    else:
        return (100*(x1**2 - x2)**2) + (x1 - 1)**2 + R*(((x1-0.4)**2 + x2**2)**2)+  R*(((x1)**2 + x2**2)**2)


def grad_center(x, epsilon, R):
    h = epsilon*np.linalg.norm(x)
    dx = (evaluate_f(x[0]+h,x[1], R)-evaluate_f(x[0]-h,x[1], R))/(2*h)
    dy = (evaluate_f(x[0],x[1]+h, R)-evaluate_f(x[0],x[1]-h, R))/(2*h)
    return np.array([dx,dy])





def gold(start, svenn, direction,epsilon_la, R):
    x0 = start[0]
    y0 = start[1]
    x = direction[0]
    y = direction[1]
    current_interval = svenn
    L = current_interval[1]-current_interval[0]
    la_1 = current_interval[0]+0.382*L
    la_2 = current_interval[0]+0.618*L
    f_la1 =evaluate_f(x0+(la_1*x),y0+(la_1*y), R)
    f_la2 =evaluate_f(x0+(la_2*x),y0+(la_2*y), R)
    while(L>epsilon_la):
        if(f_la1<f_la2):
            current_interval = [current_interval[0],la_2]
            f_la2 = f_la1
            L = current_interval[1]-current_interval[0]
            la_1 = current_interval[0]+0.382*L
            la_2 = current_interval[0]+0.618*L
            f_la1 =evaluate_f(x0+(la_1*x),y0+(la_1*y), R)
        elif(f_la1>f_la2):
            current_interval = [la_1,current_interval[1]]
            f_la1 = f_la2
            L = current_interval[1]-current_interval[0]
            la_1 = current_interval[0]+0.382*L
            la_2 = current_interval[0]+0.618*L
            f_la2 =evaluate_f(x0+(la_2*x),y0+(la_2*y), R)
    return (current_interval[0]+current_interval[1])/2


def svenn_la2(direction, start, dx, case, la0, R):
   x0 = start[0]
   y0 = start[1]
   x = direction[0]
   y = direction[1]
   nX = np.linalg.norm(start)
   f0 = evaluate_f(x0 + ((la0) * x), y0 + (
               (la0) * y), R)
   fl = evaluate_f(x0 + ((la0 - dx) * x), y0 + ((la0 - dx) * y), R)
   fr = evaluate_f(x0 + ((la0 + dx) * x), y0 + ((la0 + dx) * y), R)
   values_list = [f0]
   la_list = [la0]
   if fl > f0 and f0 > fr:
       determinator = 1
       values_list.append(fr)
       la_list.append(la0 + dx)
   elif fl < f0 and f0 < fr:
       determinator = -1
       values_list.append(fl)
       la_list.append(la0 - dx)
   elif fl > f0 and f0 < fr:
       if case == 1:
           return [la0 - dx, la0 + dx]
       else:
           return [la0 - dx, la0, la0 + dx]

   i = 1
   while (values_list[i] < values_list[i - 1]):
       la_i = la_list[i] + determinator * (2 ** i) * dx
       la_list.append(la_i)
       values_list.append(evaluate_f(x0 + ((la_i) * x), y0 + ((la_i) * y), R))
       i += 1
   last4 = [la_list[i], (la_list[i] + la_list[i - 1]) / 2, la_list[i - 1], la_list[i - 2]]
   last4_evaluated = []
   for la in last4:
       last4_evaluated.append(evaluate_f(x0 + ((la) * x), y0 + ((la) * y), R))

   ind = last4_evaluated.index(min(last4_evaluated))
   if case == 1:
       return sorted([last4[ind - 1], last4[ind + 1]])
   else:
       if ind == 1:
           last3 = [last4[0], last4[1], last4[2]]
       if ind == 2:
           last3 = [last4[1], last4[2], last4[3]]
       return last3


def dsk_powell(x, la, s, accuracy, R):
   la = sorted(la)
   x1 = la[0]
   x2 = la[1]
   x3 = la[2]
   dx = abs(x2 - x1)
   f1 = evaluate_f(x[0] + x1 * s[0], x[1] + x1 * s[1], R)
   f2 = evaluate_f(x[0] + x2 * s[0], x[1] + x2 * s[1], R)
   f3 = evaluate_f(x[0] + x3 * s[0], x[1] + x3 * s[1], R)
   x_dsk = x2 + (dx * (f1 - f3)) / (2 * (f1 - 2 * f2 + f3))
   f_dsk = evaluate_f(x[0] + x_dsk * s[0], x[1] + x_dsk * s[1], R)
   if ((x2 - x_dsk) < accuracy) and ((f2 - f_dsk) < accuracy):
       return x_dsk
   x_top = x_dsk
   end = False
   while not end:
       lis = sorted([x1, x2, x3, x_top])
       ind = lis.index(x_top)
       if ind == 0:
           lis = [lis[ind], lis[ind + 1], lis[ind + 2]]
       elif ind == 3:
           lis = [lis[ind - 2], lis[ind - 1], lis[ind]]
       else:
           lis = [lis[ind - 1], lis[ind], lis[ind + 1]]
       x1 = lis[0]
       x2 = lis[1]
       x3 = lis[2]
       f1 = evaluate_f(x[0] + x1 * s[0], x[1] + x1 * s[1], R)
       f2 = evaluate_f(x[0] + x2 * s[0], x[1] + x2 * s[1], R)
       f3 = evaluate_f(x[0] + x3 * s[0], x[1] + x3 * s[1], R)
       a1 = (f2 - f1) / (x2 - x1)
       a2 = (1 / (x3 - x2)) * ((f3 - f1) / (x3 - x1) - (f2 - f1) / (x2 - x1))
       x_top = (x1 + x2) / 2 - a1 / (2 * a2)
       f_x_top = evaluate_f(x[0] + x_top * s[0], x[1] + x_top * s[1], R)
       f_list = [f1, f2, f3, f_x_top]
       list_x = [x1, x2, x3, x_top]
       f_min = min(f_list)
       inx_min = f_list.index(f_min)
       x_min = list_x[inx_min]
       if (1 / (x3 - x2) <= 0.0001):
           end = True
       if ((f_min - f_x_top) < accuracy) and (x_min - x_top < accuracy):
           end = True
   return x_top



def Partan(x0, epsilon, h_epsilon, odnom_epsilon, case, R=0):
    x = [np.array(x0)]
    end = False
    k=1
    while not end:
        grad = grad_center(x[k-1], h_epsilon, R)
        if k%3 == 0:
            S = x[k-1] - x[k-3]
        else:
            S = -grad
        svenn = svenn_la2(S, x[k-1], 0.001, case, 0, R)
        if case == 1:
            odnom_pousk = gold(x[k-1], svenn, S, odnom_epsilon, R)
        else:
            odnom_pousk = dsk_powell(x[k-1], svenn, S, odnom_epsilon, R)
        la_opt = odnom_pousk
        x_new = x[k-1] + la_opt*(S)
        x.append(x_new)
        if(np.linalg.norm(x[k-1]-x[k])/np.linalg.norm(x[k-1])<epsilon) \
                and ((evaluate_f(x[k-1][0],x[k-1][1], R)-evaluate_f(x[k][0],x[k][1], R))<epsilon):
            break
        k +=1
    return x


end = False
x_k_list = [[-1.2, 0]]
R = 1
k = 1
while end != True:
    x_k = x_k_list[-1]
    x = Partan(x_k, 0.001, 0.01 , 0.0005, 1, R)
    print("Ітерація №{}".format(k))
    print("Кінцева точка ЗБО: ", x[-1])
    print("Кількість обчислень функції=",counter1)
    x_k_1 = x[-1]
    x_k_list.append(x_k_1)
    for dot in x:
        global_points.append(dot)
    if ((np.linalg.norm(np.array(x_k_1) - np.array(x_k)))/np.linalg.norm(x_k) < 0.001):
        end = True
        break
    R *= 10
    k +=1


x_list = []
y_list = []
for lis in x_k_list:
   x_list.append(lis[0])
   y_list.append(lis[1])
dots_x = []
dots_y = []
for lis in global_points:
   dots_x.append(lis[0])
   dots_y.append(lis[1])
circle1 = plt.Circle((0,0),1,fill=True)
circle2 = plt.Circle((0.4,0),0.3,fill=True,color='white')
fig, ax = plt.subplots()
ax.add_artist(circle1)
ax.add_artist(circle2)
plt.plot(x_list,y_list, color = "purple")
plt.plot(dots_x,dots_y, color = "green")
plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(x_list,y_list, color = "purple")
plt.scatter([1],[1], color = "red")
green_patch = mpatches.Patch(color='green', label='Шлях пошуку ЗБО')
purple_patch = mpatches.Patch(color='purple', label='Перехід від одного розв\'язку ЗБО до іншого')
red_patch = mpatches.Patch(color='red', label='Справжній мінімум')
plt.legend(handles=[green_patch,purple_patch,red_patch])
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.show()


