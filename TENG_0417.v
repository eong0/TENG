`include "disciplines.h"

module TENG(Q,X,Icap,Vteng);

   input [3:0] Q; 
   input [3:0] X;
   
   electrical Icap,Vteng;
 
   real e0 = 8.85e-14 * 1e+15;
   real sig = 50;
   real d0 = 4;//thickness
   real S = 10;//area
   real dt = 1e-6 * 2e+18;

   real  	  Cap=0;
   real 	  Vcap=0;

   analog begin  
      Cap = S*e0/(d0 + X);
      Vcap = -Q/Cap;
      V(Vteng) <+ Vcap + sig*X/e0;
      I(Icap) <+ -ddt(Vcap/dt)*Cap;
   end // UNMATCHED !!
   
endmodule // TENG 
