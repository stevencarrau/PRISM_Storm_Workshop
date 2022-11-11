mdp

const int N=5;
const int xMAX = N-1;
const int yMAX = N-1;
const int dxMAX = xMAX;
const int dyMAX = yMAX;
const int dxMIN = 0;
const int dyMIN = 0;
const double slippery = 0.1;


formula westenabled = dx != dxMIN;
formula eastenabled = dx != dxMAX;
formula northenabled = dy != dyMIN;
formula southenabled = dy != dyMAX;
formula done = dx = dxMAX & dy = dyMAX;
formula crash = (dx < 4 & dy=1) |  (dx >1 & dy=3);

module master
    [north] !done -> true;
    [south] !done -> true;
    [east]  !done -> true;
    [west]  !done -> true;
    [done] done -> true;	
endmodule


module drone
    dx : [dxMIN..dxMAX] init dxMIN;
    dy : [dyMIN..dyMAX] init dyMIN;
    [west] westenabled -> (1-slippery): (dx'=max(dx-1,dxMIN)) + slippery: (dx'=max(dx-2,dxMIN));
    [east] eastenabled -> (1-slippery): (dx'=min(dx+1,dxMAX)) + slippery: (dx'=min(dx+2,dxMAX));
    [south]  southenabled -> (1-slippery): (dy'=min(dy+1,dyMAX)) + slippery: (dy'=min(dy+2,dyMAX));
    [north]  northenabled -> (1-slippery): (dy'=max(dy-1,dyMIN)) + slippery: (dy'=max(dy-2,dyMIN));
endmodule

rewards
    [north] true : 1;
    [south] true : 1;
    [east] true : 1;
    [west] true : 1;
    crash : 100;
endrewards

label "goal" = done;
label "crash" = crash;
