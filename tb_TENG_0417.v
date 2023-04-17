`timescale 1ns/1ns

module tb_TENG1(Q,X,Icap,Vteng);
   
   output reg [3:0] Q;
   
   output reg [3:0] X;
   
   output reg  Icap,Vteng;
   
   //real Q = 0;
   //real X = 0;
   //real Icap;
   //real Vteng;
   real t=0;
   real dt = 1e-6;
   //real Q = 12;
   

   initial begin
      Q=0;
      X = 0;
      Icap = 0;
      Vteng = 0;
      
      #10
	Q=120;
       X =20;
      #10
        X=70;
      #10
        X=40;
      #10
	X=20;
      #10
        X=90;
      #10
	X=10;
      #10
	X=40;
      #10
	X=5;
      #10
	X=80;
      #10;
      
      
   end
   
endmodule // tb_TENG
