`timescale 1ns / 1ps
typedef integer TYPE;

module hadamard #(parameter VEC_LEN = 9)
    (input TYPE params[VEC_LEN-1:0],
     input TYPE weights[VEC_LEN-1:0],
     output TYPE res[VEC_LEN-1:0]);
     
    genvar i;
    generate for (i = 0; i < VEC_LEN; i = i+1) begin
        assign res[i] = params[i]*weights[i];
    end endgenerate
endmodule: hadamard

module conv_ind #(parameter CONV_SIZE = 9)
    (input TYPE params[CONV_SIZE - 1:0],
     input TYPE weights[CONV_SIZE - 1:0],
     input TYPE bias,
     output TYPE res);
     
    TYPE vec_product[CONV_SIZE - 1:0];
    hadamard #(CONV_SIZE) h(params, weights, vec_product);
    
    assign res = bias + vec_product.sum;
endmodule: conv_ind

//only for stride 2 pools
module pool_fn (input TYPE x1, input TYPE x2, input TYPE x3, input TYPE x4, output TYPE y);   
    TYPE z1; 
    assign z1 = (x1 > x2) ? x1 : x2;
    TYPE z2;
    assign z2 = (x3 > x4) ? x3 : x4; 
    assign y = (z1 > z2) ? z1 : z2;
endmodule: pool_fn

//stride 2 only!
module pool #(parameter H = 8, parameter W = 8)
    (input TYPE arr[H-1:0][W-1:0],
     output TYPE pooled[H/2-1:0][W/2-1:0]);
    genvar i, j;
    generate
        for (i = 0; i < H; i = i+2) begin
            for(j = 0; j < W; j = j+2) begin
                pool_fn F(arr[i][j],
                          arr[i+1][j],
                          arr[i][j+1],
                          arr[i+1][j+1],
                          pooled[i/2][j/2]);
            end
        end
    endgenerate
endmodule: pool

module pool3d #(parameter H = 8, parameter W = 8, parameter D = 3)
    (input TYPE arr[D-1:0][H-1:0][W-1:0],
     output TYPE pooled[D-1:0][H/2-1:0][W/2-1:0]);
     
     genvar chan;
     generate
     for (chan = 0; chan < D; chan ++) begin
        pool #(H, W) P(arr[chan], pooled[chan]);     
     end
     endgenerate
     
endmodule: pool3d

module pad #(parameter H = 8, parameter W = 8, parameter M = 1)
    (input TYPE arr[H-1:0][W-1:0],
     output TYPE padded[H-1+2*M:0][W-1+2*M:0]);

    genvar i, j;
    generate
        for(i = 0; i < H+2*M; i++) begin
            for(j = 0; j < W+2*M; j++) begin
                if(i < M | i >= H+M | j < M | j >= W+M)
                    assign padded[i][j] = 0;
                else 
                    assign padded[i][j] = arr[i-M][j-M];                   
            end
        end    
    endgenerate
endmodule: pad

module pad3d #(parameter H = 8, parameter W = 8, parameter D = 3, parameter M = 1)
    (input TYPE arr[D-1:0][H-1:0][W-1:0],
     output TYPE padded[D-1:0][H-1+2*M:0][W-1+2*M:0]);
     
    genvar chan;
    generate
    for (chan = 0; chan < D; chan ++) begin
        pad #(H, W, M) P(arr[chan], padded[chan]);
    end
    endgenerate
     
endmodule: pad3d

module conv #(parameter ARR_WIDTH = 8, parameter ARR_HEIGHT = 8, parameter CONV_SIZE = 3, parameter CONV_MARGIN = 1)
    (input TYPE arr[ARR_HEIGHT-1:0][ARR_WIDTH-1:0],
     input TYPE weights[CONV_SIZE*CONV_SIZE - 1:0],
     input TYPE bias,
     output TYPE res[ARR_HEIGHT-CONV_SIZE:0][ARR_WIDTH-CONV_SIZE:0]);
     
    genvar i, j;
    generate for (i = CONV_MARGIN; i < ARR_HEIGHT-CONV_MARGIN; i = i+1) begin
        for (j = CONV_MARGIN; j < ARR_WIDTH-CONV_MARGIN; j = j+1) begin
            TYPE conv_ind_input[8:0];
            assign conv_ind_input = {arr[i-1][j-1], arr[i-1][j], arr[i-1][j+1], arr[i][j-1], arr[i][j], arr[i][j+1], arr[i+1][j-1], arr[i+1][j], arr[i+1][j+1]};
            conv_ind #(CONV_SIZE*CONV_SIZE) C(conv_ind_input, weights, bias, res[i-CONV_MARGIN][j-CONV_MARGIN]);
        end
    end endgenerate
endmodule: conv

module conv_ind_channel #(parameter ARR_WIDTH = 8, parameter ARR_HEIGHT = 8, parameter CONV_SIZE = 3, parameter CONV_MARGIN = 1, parameter NUM_CHANNELS = 3)
    (input TYPE arr[NUM_CHANNELS-1:0][ARR_HEIGHT-1:0][ARR_WIDTH-1:0],
     input TYPE weights[CONV_SIZE*CONV_SIZE*NUM_CHANNELS - 1:0],
     input TYPE bias,
     output TYPE res[ARR_HEIGHT-CONV_SIZE:0][ARR_WIDTH-CONV_SIZE:0]);
     
    genvar i, j, k, offi, offj;
    generate 
        for (i = CONV_MARGIN; i < ARR_HEIGHT - CONV_MARGIN; i = i+1) begin
            for (j = CONV_MARGIN; j < ARR_WIDTH - CONV_MARGIN; j = j+1) begin
                TYPE conv_ind_input[CONV_SIZE*CONV_SIZE*NUM_CHANNELS-1:0];
            
                for (k = 0; k < NUM_CHANNELS; k = k+1) begin
                    for(offi = -CONV_MARGIN; offi <= CONV_MARGIN; offi ++) begin
                        for(offj = -CONV_MARGIN; offj <= CONV_MARGIN; offj++) begin
                            assign conv_ind_input[k*CONV_SIZE*CONV_SIZE + (offi+CONV_MARGIN)*CONV_SIZE + (offj+CONV_MARGIN)] = arr[k][i+offi][j+offj];
                        end                
                    end
                end
                conv_ind #(CONV_SIZE*CONV_SIZE*NUM_CHANNELS) c(conv_ind_input, weights, bias, res[i-CONV_MARGIN][j-CONV_MARGIN]);
            end
        end 
    endgenerate
endmodule: conv_ind_channel

module conv_channels #(parameter ARR_WIDTH = 8, parameter ARR_HEIGHT = 8, parameter CONV_SIZE = 3, parameter CONV_MARGIN = 1, parameter NUM_CHANNELS = 3, parameter NUM_KERNELS = 3)
    (input TYPE arr[NUM_CHANNELS-1:0][ARR_HEIGHT-1:0][ARR_WIDTH-1:0],
     input TYPE weights[NUM_KERNELS-1:0][CONV_SIZE*CONV_SIZE*NUM_CHANNELS - 1:0],
     input TYPE bias[NUM_KERNELS-1:0],
     output TYPE res[NUM_KERNELS-1:0][ARR_HEIGHT-CONV_SIZE:0][ARR_WIDTH-CONV_SIZE:0]);

    genvar i;
    generate for (i = 0; i < NUM_KERNELS; i = i+1) begin
        conv_ind_channel #(ARR_WIDTH, ARR_HEIGHT, CONV_SIZE, CONV_MARGIN, NUM_CHANNELS) c(arr, weights[i], bias[i], res[i]);
    end endgenerate
endmodule: conv_channels

/*
module run #(parameter ARR_WIDTH = 8, parameter ARR_HEIGHT = 8, parameter CONV_SIZE = 3, parameter CONV_MARGIN = 1)
    (input TYPE arr[ARR_HEIGHT-1:0][ARR_WIDTH-1:0],
     input TYPE weights[CONV_SIZE*CONV_SIZE-1:0],
     input TYPE bias,
     output TYPE res[ARR_HEIGHT-1:0][ARR_WIDTH-1:0]);
     
    byte pad_arr[ARR_HEIGHT+2*CONV_MARGIN-1:0][ARR_WIDTH+2*CONV_MARGIN-1:0];
    pad #(ARR_HEIGHT, ARR_WIDTH, CONV_MARGIN) p(arr, pad_arr);
    
    byte conv_arr[ARR_HEIGHT-1:0][ARR_WIDTH-1:0];
    conv #(ARR_WIDTH, ARR_HEIGHT, CONV_SIZE, CONV_MARGIN) c(pad_arr, weights, bias, conv_arr);
    
    relu_arr #(ARR_WIDTH, ARR_HEIGHT) r(conv_arr, res);
endmodule: run
*/

module run_channels #(parameter ARR_WIDTH = 8, parameter ARR_HEIGHT = 8, parameter CONV_SIZE = 3, parameter CONV_MARGIN = 1, parameter NUM_CHANNELS = 3, parameter NUM_KERNELS = 3, parameter DO_RELU = 1)
    (input TYPE arr[NUM_CHANNELS-1:0][ARR_HEIGHT-1:0][ARR_WIDTH-1:0],
     input TYPE weights[NUM_KERNELS-1:0][CONV_SIZE*CONV_SIZE*NUM_CHANNELS-1:0],
     input TYPE bias[NUM_KERNELS-1:0],
     output TYPE res[NUM_KERNELS-1:0][ARR_HEIGHT-1:0][ARR_WIDTH-1:0]);
     
    TYPE pad_arr[NUM_CHANNELS-1:0][ARR_HEIGHT+2*CONV_MARGIN-1:0][ARR_WIDTH+2*CONV_MARGIN-1:0];
    pad3d #(ARR_HEIGHT, ARR_WIDTH, NUM_CHANNELS, CONV_MARGIN) p(arr, pad_arr);
    
    TYPE conv_arr[NUM_KERNELS-1:0][ARR_HEIGHT-1:0][ARR_WIDTH-1:0];
    conv_channels #(ARR_WIDTH+CONV_MARGIN*2, ARR_HEIGHT+CONV_MARGIN*2, CONV_SIZE, CONV_MARGIN, NUM_CHANNELS, NUM_KERNELS) c(pad_arr, weights, bias, conv_arr);
    
    if(DO_RELU)
        relu_arr_3d #(ARR_WIDTH, ARR_HEIGHT, NUM_KERNELS) r(conv_arr, res);
    else
        assign res = conv_arr;
        
endmodule: run_channels

module relu (input TYPE x, output TYPE y);
    assign y[31:22] = 10'b0;
    assign y[21:0] = x[31:10];
endmodule: relu

module relu_arr #(parameter ARR_WIDTH = 8, parameter ARR_HEIGHT = 8)
    (input TYPE arr[ARR_HEIGHT-1:0][ARR_WIDTH-1:0],
     output TYPE res[ARR_HEIGHT-1:0][ARR_WIDTH-1:0]);
     
    genvar i, j;
    generate for (i = 0; i < ARR_HEIGHT; i = i+1) begin
        for (j = 0; j < ARR_WIDTH; j = j+1) begin
            relu r(arr[i][j], res[i][j]);
        end
    end endgenerate
endmodule: relu_arr

module relu_arr_3d #(parameter ARR_WIDTH = 8, parameter ARR_HEIGHT = 8, parameter NUM_KERNELS = 3)
    (input TYPE arr[NUM_KERNELS-1:0][ARR_HEIGHT-1:0][ARR_WIDTH-1:0],
     output TYPE res[NUM_KERNELS-1:0][ARR_HEIGHT-1:0][ARR_WIDTH-1:0]);
     
    genvar i;
    generate for (i = 0; i < NUM_KERNELS; i = i+1) begin
        relu_arr #(ARR_WIDTH, ARR_HEIGHT) r(arr[i], res[i]);
    end endgenerate
endmodule: relu_arr_3d


module network #(parameter inH = 32, parameter inW = 32, parameter inD = 3, outD = 10)
    (input TYPE img[inD-1:0][inH-1:0][inW-1:0],
     input TYPE weights1[8-1:0][3*3*3-1:0],
     input TYPE bias1[8-1:0],
     input TYPE weights2[16-1:0][3*3*8-1:0],
     input TYPE bias2[16-1:0],
     input TYPE weights3[32-1:0][3*3*16-1:0],
     input TYPE bias3[32-1:0],
     input TYPE weights4[16-1:0][1*1*32-1:0],
     input TYPE bias4[16-1:0],
     input TYPE weights5[32-1:0][3*3*16-1:0],
     input TYPE bias5[32-1:0],
     input TYPE weights6[16-1:0][1*1*32-1:0],
     input TYPE bias6[16-1:0],
     input TYPE weights7[32-1:0][3*3*16-1:0],
     input TYPE bias7[32-1:0],
     input TYPE weights8[16-1:0][1*1*32-1:0],
     input TYPE bias8[16-1:0],
     input TYPE weights9[outD-1:0][1*1*16-1:0],
     input TYPE bias9[outD-1:0],
     output TYPE pred[outD-1:0]);
    
    TYPE res1[8-1:0][inH-1:0][inW-1:0];
    run_channels #(inW, inH, 3, 1, inD, 8) r1(img, weights1, bias1, res1);
    
    TYPE res1_pooled[8-1:0][inH/2 - 1:0][inW/2 - 1:0];
    pool3d #(inH, inW, 8) p1(res1, res1_pooled);
    
    TYPE res2[16-1:0][inH/2-1:0][inW/2-1:0];
    run_channels #(inW/2, inH/2, 3, 1, 8, 16) r2(res1_pooled, weights2, bias2, res2);
    
    TYPE res2_pooled[16-1:0][inH/2/2 - 1:0][inW/2/2 - 1:0];
    pool3d #(inH/2, inW/2, 16) p2(res2, res2_pooled);
    
    TYPE res3[32-1:0][inH/2/2 - 1:0][inW/2/2 - 1:0];
    run_channels #(inW/2/2, inH/2/2, 3, 1, 16, 32) r3(res2_pooled, weights3, bias3, res3);
    
    TYPE res3_pooled[32-1:0][inH/2/2/2-1:0][inW/2/2/2-1:0];
    pool3d #(inH/2/2, inW/2/2, 32) p3(res3, res3_pooled);
    
    TYPE res4[16-1:0][inH/2/2/2 - 1:0][inW/2/2/2 - 1:0];
    run_channels #(inW/2/2/2, inH/2/2/2, 1, 0, 32, 16) r4(res3_pooled, weights4, bias4, res4);
    
    TYPE res5[32-1:0][inH/2/2/2 - 1:0][inW/2/2/2 - 1:0];
    run_channels #(inW/2/2/2, inH/2/2/2, 3, 1, 16, 32) r5(res4, weights5, bias5, res5);
    
    TYPE res5_pooled[32-1:0][inH/2/2/2/2-1:0][inW/2/2/2/2-1:0];
    pool3d #(inH/2/2/2, inW/2/2/2, 32) p5(res5, res5_pooled);
    
    TYPE res6[16-1:0][inH/2/2/2/2 - 1:0][inW/2/2/2/2 - 1:0];
    run_channels #(inW/2/2/2/2, inH/2/2/2/2, 1, 0, 32, 16) r6(res5_pooled, weights6, bias6, res6);
        
    TYPE res7[32-1:0][inH/2/2/2/2 - 1:0][inW/2/2/2/2 - 1:0];
    run_channels #(inW/2/2/2/2, inH/2/2/2/2, 3, 1, 16, 32) r7(res6, weights7, bias7, res7);
    
    TYPE res7_pooled[32-1:0][inH/2/2/2/2/2-1:0][inW/2/2/2/2/2-1:0];
    pool3d #(inH/2/2/2/2, inW/2/2/2/2, 32) p7(res7, res7_pooled);
    
    TYPE res8[16-1:0][inH/2/2/2/2/2 - 1:0][inW/2/2/2/2/2 - 1:0];
    run_channels #(inW/2/2/2/2/2, inH/2/2/2/2/2, 1, 0, 32, 16) r8(res7_pooled, weights8, bias8, res8);  
    
    TYPE pred_unshaped[outD-1:0][0:0][0:0];
    run_channels #(inW/2/2/2/2/2, inH/2/2/2/2/2, 1, 0, 16, 10, 0) r11(res8, weights9, bias9, pred_unshaped);
    
    genvar chan;
    generate 
        for(chan = 0; chan < outD; chan++) begin
            assign pred[chan] = pred_unshaped[chan][0][0];
        end
    endgenerate
    
endmodule: network

module test_conv_ind_channel();
    // 4x4x3 image, 3x3x3 kernel
    TYPE weights[27-1:0] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    TYPE arr[3-1:0][4-1:0][4-1:0];
    TYPE bias;
    TYPE res[1:0][1:0];
    
    genvar i, j, k;
    generate for (i = 0; i < 3; i = i+1) begin
        for (j = 0; j < 4; j = j+1) begin
            for (k = 0; k < 4; k = k+1) begin
                assign arr[i][j][k] = 16*i+4*j+k;
            end
        end
    end endgenerate

    assign bias = 2;
    
    conv_ind_channel #(4, 4, 3, 1, 3) c(arr, weights, bias, res);
endmodule: test_conv_ind_channel

module test_conv();
    TYPE arr[8-1:0][8-1:0];
    TYPE weights[3*3 - 1:0];
    TYPE bias;
    TYPE res[8-3:0][8-3:0];
    
    conv #(8, 8, 3, 1) C(.*);
    
    genvar i, j;
    generate for (i = 0; i < 8; i = i+1) begin
        for (j = 0; j < 8; j = j + 1) begin
            assign arr[i][j] = i+j;
        end
    end endgenerate
    
    generate for (i = 0; i < 9; i = i+1) begin
        assign weights[i] = 1;
    end endgenerate
    
    assign bias = 3;
    
endmodule

module test_conv_ind();
    TYPE params[8:0];
    TYPE weights[8:0];
    TYPE bias;
    TYPE res;
    
    conv_ind #(3) C(.*);
    
    genvar i;
    generate for (i = 0; i < 9; i = i+1) begin
        assign params[i] = i;
        assign weights[i] = 1;
    end endgenerate
    
    assign bias = 2;
endmodule

module top2 #(parameter outD = 10, parameter N = 1)
    (output TYPE pred[outD-1:0]);
    
    genvar i, j, k;
    
    TYPE img0[3072-1:0];
    TYPE img1[3072-1:0];
    TYPE img2[3072-1:0];
    TYPE img3[3072-1:0];
    TYPE img4[3072-1:0];
    TYPE img5[3072-1:0];
    TYPE img6[3072-1:0];
    TYPE img7[3072-1:0];
    TYPE img8[3072-1:0];
    TYPE img9[3072-1:0];
    
    cifar #(0) I0(img0);
    cifar #(1) I1(img1);
    cifar #(2) I2(img2);
    cifar #(3) I3(img3);
    cifar #(4) I4(img4);
    cifar #(5) I5(img5);
    cifar #(6) I6(img6);
    cifar #(7) I7(img7);
    cifar #(8) I8(img8);
    cifar #(9) I9(img9);
    
    TYPE img_rs[3-1:0][32-1:0][32-1:0];

    TYPE img_rs0[3-1:0][32-1:0][32-1:0];
    TYPE img_rs1[3-1:0][32-1:0][32-1:0];
    TYPE img_rs2[3-1:0][32-1:0][32-1:0];
    TYPE img_rs3[3-1:0][32-1:0][32-1:0];
    TYPE img_rs4[3-1:0][32-1:0][32-1:0];
    TYPE img_rs5[3-1:0][32-1:0][32-1:0];
    TYPE img_rs6[3-1:0][32-1:0][32-1:0];
    TYPE img_rs7[3-1:0][32-1:0][32-1:0];
    TYPE img_rs8[3-1:0][32-1:0][32-1:0];
    TYPE img_rs9[3-1:0][32-1:0][32-1:0];
    
    generate for (i = 0; i < 3; i = i+1) begin
        for (j = 0; j < 32; j = j+1) begin
            for (k = 0; k < 32; k = k+1) begin
                assign img_rs0[i][j][k] = img0[32*32*i + 32*j + k];
                assign img_rs1[i][j][k] = img1[32*32*i + 32*j + k];
                assign img_rs2[i][j][k] = img2[32*32*i + 32*j + k];
                assign img_rs3[i][j][k] = img3[32*32*i + 32*j + k];
                assign img_rs4[i][j][k] = img4[32*32*i + 32*j + k];
                assign img_rs5[i][j][k] = img5[32*32*i + 32*j + k];
                assign img_rs6[i][j][k] = img6[32*32*i + 32*j + k];
                assign img_rs7[i][j][k] = img7[32*32*i + 32*j + k];
                assign img_rs8[i][j][k] = img8[32*32*i + 32*j + k];
                assign img_rs9[i][j][k] = img9[32*32*i + 32*j + k];
            end
        end
    end
    
    TYPE weights1[8*3*3*3-1:0];
    TYPE bias1[8-1:0];
    TYPE weights2[16*3*3*8-1:0];
    TYPE bias2[16-1:0];
    TYPE weights3[32*3*3*16-1:0];
    TYPE bias3[32-1:0];
    TYPE weights4[16*1*1*32-1:0];
    TYPE bias4[16-1:0];
    TYPE weights5[32*3*3*16-1:0];
    TYPE bias5[32-1:0];
    TYPE weights6[16*1*1*32-1:0];
    TYPE bias6[16-1:0];
    TYPE weights7[32*3*3*16-1:0];
    TYPE bias7[32-1:0];
    TYPE weights8[16*1*1*32-1:0];
    TYPE bias8[16-1:0];
    TYPE weights9[outD*1*1*16-1:0];
    TYPE bias9[outD-1:0];
    
    parameters p(weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5, weights6, bias6, weights7, bias7, weights8, bias8, weights9, bias9);
    
    TYPE weights1_rs[8-1:0][3*3*3-1:0];
    for (i = 0; i < 8; i = i+1) begin
        for (j = 0; j < 3*3*3; j = j+1) begin
            assign weights1_rs[i][j] = weights1[3*3*3*i + j];
        end
    end
    
    TYPE weights2_rs[16-1:0][3*3*8-1:0];
    for (i = 0; i < 16; i = i+1) begin
        for (j = 0; j < 3*3*8; j = j+1) begin
            assign weights2_rs[i][j] = weights2[3*3*8*i + j];
        end
    end
    
    TYPE weights3_rs[32-1:0][3*3*16-1:0];
    for (i = 0; i < 32; i = i+1) begin
        for (j = 0; j < 3*3*16; j = j+1) begin
            assign weights3_rs[i][j] = weights3[3*3*16*i + j];
        end
    end
    
    TYPE weights4_rs[16-1:0][1*1*32-1:0];
    for (i = 0; i < 16; i = i+1) begin
        for (j = 0; j < 1*1*32; j = j+1) begin
            assign weights4_rs[i][j] = weights4[1*1*32*i + j];
        end
    end
        
    TYPE weights5_rs[32-1:0][3*3*16-1:0];
    for (i = 0; i < 32; i = i+1) begin
        for (j = 0; j < 3*3*16; j = j+1) begin
            assign weights5_rs[i][j] = weights5[3*3*16*i + j];
        end
    end
        
    TYPE weights6_rs[16-1:0][1*1*32-1:0];
    for (i = 0; i < 16; i = i+1) begin
        for (j = 0; j < 1*1*32; j = j+1) begin
            assign weights6_rs[i][j] = weights6[1*1*32*i + j];
        end
    end
        
    TYPE weights7_rs[32-1:0][3*3*16-1:0];
    for (i = 0; i < 32; i = i+1) begin
        for (j = 0; j < 3*3*16; j = j+1) begin
            assign weights7_rs[i][j] = weights7[3*3*16*i + j];
        end
    end
        
    TYPE weights8_rs[16-1:0][1*1*32-1:0];
    for (i = 0; i < 16; i = i+1) begin
        for (j = 0; j < 1*1*32; j = j+1) begin
            assign weights8_rs[i][j] = weights8[1*1*32*i + j];
        end
    end
        
    TYPE weights9_rs[outD-1:0][1*1*16-1:0];
    for (i = 0; i < outD; i = i+1) begin
        for (j = 0; j < 1*1*16; j = j+1) begin
            assign weights9_rs[i][j] = weights9[1*1*16*i + j];
        end
    end endgenerate
        
    //TYPE pred[outD-1:0];
    network #(32, 32, 3, 10) n(img_rs, weights1_rs, bias1, weights2_rs, bias2, weights3_rs, bias3, weights4_rs, bias4, weights5_rs, bias5, weights6_rs, bias6, weights7_rs, bias7, weights8_rs, bias8, weights9_rs, bias9, pred);
    
    initial begin
        #100
        img_rs = img_rs0;
        #100
        img_rs = img_rs1;
        #100
        img_rs = img_rs2;
        #100
        img_rs = img_rs3;
        #100
        img_rs = img_rs4;
        #100
        img_rs = img_rs5;
        #100
        img_rs = img_rs6;
        #100
        img_rs = img_rs7;
        #100
        img_rs = img_rs8;
        #100
        img_rs = img_rs9;
    end
    
    //TYPE queue[$];
    //assign queue = pred.max;
    //TYPE max;
    //assign max = queue.pop_front();
    //assign res = pred.find_first_index(max).pop_front();
    
endmodule: top2
