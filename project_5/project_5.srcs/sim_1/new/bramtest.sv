`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/25/2017 05:00:57 PM
// Design Name: 
// Module Name: bramtest
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module bramtest();
    //write?
    logic clka; 
    logic [0:0]wea;
    logic [15:0]addra;
    logic [31:0]dina;
    logic [31:0]douta;
        
    blk_mem_gen_0 B(.*);

    initial begin
        clka = 1'b0;
        forever #10 clka = ~clka;
    end
    
    initial begin
        addra = 16'b0;
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        wea = 1'b1;
        dina = 32'd23;
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        $display("out=", douta);
        wea = 0'b0;
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        @(posedge clka);
        $display("out=", douta);
    end
    
endmodule
