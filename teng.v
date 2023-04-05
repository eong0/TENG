`include "disciplines.h"

module teng(en,Q,X,Icap,Vteng);
   parameter real e0 = 8.85e-14;
   //parameter real er1 = 2.2;
   parameter real sig = 50;
   parameter real d0 = 4; //thickness
   parameter real S = 10; //area

   real 	  Cap;
   real 	  Vcap;
   
   input 	  en,Q,x;
   output 	  Icap,Vteng;

   analog begin @(cross (V(en)-1.5*vdd)) //for resetting
      Cap = S*e0/(d0 + X);
      Vcap = -Q/Cap;
      Vteng = Vcap + sig*X/e0;
      Icap = (d(Vcap)/dt)*Cap;
   end
      
endmodule // teng
