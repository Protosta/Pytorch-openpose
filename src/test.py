import numpy as np
import math
import cv2 as cv
import os

neck=[144.0,155.0,161.0,173.0,152.0,142.0,151.0,115.0,137.0,
 155.0,133.0,131.0,115.0,126.0,159.0,146.0,173.0,156.0,
 149.0,154.0,147.0,136.0,155.0,162.0,145.0,137.0,143.0,
 155.0,141.0,147.0,163.0,139.0,141.0,154.0,140.0,161.0,
 150.0,170.0,158.0,150.0,151.0,116.0,130.0,156.0,161.0]
body=[881.0,959.0,912.0,900.0,964.0,892.0,879.0,811.0,869.0,
 892.0,919.0,931.0,811.0,864.0,883.0,874.0,863.0,831.0,
 851.0,860.0,842.0,852.0,880.0,981.0,863.0,829.0,862.0,
 875.0,907.0,965.0,822.0,880.0,876.0,908.0,795.0,867.0,
 852.0,924.0,909.0,859.0,894.0,932.0,819.0,853.0,951.0]
shoulder=[243.0,274.0,248.0,232.0,254.0,232.0,218.0,204.0,236.0,
 248.0,237.0,254.0,204.0,224.0,226.0,219.0,242.0,237.0,
217.0,218.0,228.0,241.0,239.0,242.0,238.0,204.0,227.0,
 243.0,247.0,244.0,226.0,225.0,231.0,243.0,201.0,225.0,
 239.0,236.0,233.0,227.0,247.0,234.0,210.0,187.0,229.0]
leg=[538.0,564.0,523.0,534.0,602.0,568.0,536.0,491.0,539.0,
 531.0,546.0,550.0,491.0,517.0,542.0,533.0,497.0,481.0,
 521.0,522.0,514.0,502.0,497.0,584.0,508.0,510.0,497.0,
 510.0,537.0,582.0,482.0,525.0,545.0,535.0,470.0,510.0,
 502.0,560.0,564.0,487.0,517.0,539.0,520.0,516.0,575.0]

neck_test = [162.0,137.0,150.0,170.0,150.0,150.0,171.0,154.0,155.0,148.0,171.0]
body_test = [850.0,812.0,852.0,863.0,919.0,956,814.0,869.0,847.0,895.0,969.0]
shoulder_test = [238.0,205.0,234.0,240.0,250.0,272.0,246.0,248.0,277.0,282.0,225.0]
leg_test = [472.0,491.0,508.0,508.0,581.0,592.0,472.0,528.0,511.0,548.0,602.0]
#if __name__ == "__main__":
variance_min = 1000000
variance_number = -1
j=0
for i in range(0, len(body)):
 variance = abs(body_test[j] - body[i]) + abs(neck_test[j] - neck[i])+abs(shoulder_test[j] - shoulder[i])+abs(leg_test[j] - leg[i])
 if (variance_min > variance):
  variance_number = i
  variance_min = variance
print(variance_min)
print(variance_number)

