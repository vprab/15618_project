`timescale 1ns / 1ps

module testbench();
    
    logic clk = 0;
    logic [7:0] switch;
    logic [7:0] led;
    project_5 main(.*);
    
    /*
    byte table8 [8-1:0] = {0, 1, 2, 3, 4, 5, 6, 7};
    byte table64 [64-1:0] = {table8, table8, table8, table8, table8, table8, table8, table8};
    byte table512 [512-1:0] = {table64, table64, table64, table64, table64, table64, table64, table64};
    byte table4096 [4096-1:0] = {table512, table512, table512, table512, table512, table512, table512, table512};
    byte table32768 [32768-1:0] = {table4096, table4096, table4096, table4096, table4096, table4096, table4096, table4096};
    byte hugetable [65536-1:0] = {table32768, table32768};
    */
    
    initial begin
        forever #10 clk = ~clk;
    end
    
    initial begin
        #30
        switch[7:4] = 4'b1111;
        switch[3:0] = 4'b0000;
        #1000000000
        $finish;
    end
 
endmodule