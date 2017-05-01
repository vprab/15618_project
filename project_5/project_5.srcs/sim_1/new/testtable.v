`timescale 1ns / 1ps

module testtable();
    logic clk;
    logic [7:0] switch;
    logic [7:0] led;
      
    logic reset;
    int idx;
    logic dowrite;
    byte inelem;
    byte outelem;
    
    treearray #(65536) T(.*);
    
    initial begin
        reset = 0;
        dowrite = 0;
        inelem = 0;
        clk = 0;
        forever #10 clk = ~clk;
    end
    
    initial begin
        #30
        reset = 1;
        #20
        reset = 0;
        #20
        idx = 10000;
        inelem = 8'b10001000;
        dowrite = 1;
        #20
        $display("elem is: %B", outelem);
        //assert (outelem == 8'b10001000);
        dowrite = 0;
        #20
        idx = 54;
        $display("elem is: %B", outelem);
        //assert (outelem == 0);
        #20
        reset = 1;
        #20
        reset = 0;
        idx = 10000;
        $display("elem is: %B", outelem);
        //assert (outelem == 0);
        #100
        $finish;
    end
endmodule