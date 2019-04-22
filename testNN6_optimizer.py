import tensorflow as tf

# How to increase the speed of training?
# stochastic Gradient Descent(SGD)
# normal: W += -lr*delta_X
# Momentum: m = b1*m - lr*delta_x, W += m
# AdaGrad: v += dx^2, W += -lr*dx/√v
# RMSProp: v = b1*v + (1-b1)*dx^2, W += -lr*dx/√v
# Adam: m = b1*m + (1-b1)*dx, v = b2*v + (1-b2)*dx^2, W += -lr*m/√v
# Adam通常是最好的加速方法
