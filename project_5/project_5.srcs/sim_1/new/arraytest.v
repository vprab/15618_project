`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/20/2017 05:53:44 PM
// Design Name: 
// Module Name: arraytest
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


module arraytest();

    logic clk;
    
    byte image [3071:0]; 
    cifar #(5) I(image);

    byte firstbyte;
    assign firstbyte = image[0];

    initial begin
        clk = 0;
        forever #10 clk = ~clk;
    end
    
endmodule
