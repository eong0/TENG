// VerilogA for eon, pulse_gen, veriloga

`include "constants.vams"
`include "disciplines.vams"


module sin_wave_gen (clk);

output clk;
electrical clk;

real clk_var1;        
real per1,per2,per3;
parameter real tt=0.01n;      

analog begin 
	@ (initial_step)
		begin
			per1 = $abstime + 2n;
		end

	@(timer(0,per1)) begin
		clk_var1= sin($abstime*100);
		per2 = per1 + 2n;
	
		@(timer(per1,per2)) begin
			clk_var1 = -sin($abstime*100);
			per3 = per2 + 2n;

			@(timer(per2,per3)) begin
				clk_var1 = sin($abstime*100);
			end
		end
	end

		V(clk) <+ transition(clk_var1,0,tt);
  
  end
endmodule
