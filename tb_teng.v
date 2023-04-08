module tb_teng;
   real Q = 0;
   real X = 0;
   real Icap;
   real Vteng;
   real t;
   real dt = 1e-6;

   teng teng(.Q(Q), .X(X), .Icap(Icap), .Vteng(Vteng));

   initial begin
      Q = 0;
      X = 0;

      repeat(1000000)
endmodule // tb_teng
