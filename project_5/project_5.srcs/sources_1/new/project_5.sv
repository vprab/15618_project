`timescale 1ns / 1ps
typedef byte TYPE;

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

   /*
    generate
    if (CONV_SIZE == 1) begin
        assign res = bias+vec_product[0];
    end else if (CONV_SIZE == 9) begin
        assign res = (bias+
                      vec_product[0]+vec_product[1]+vec_product[2]+
                      vec_product[3]+vec_product[4]+vec_product[5]+
                      vec_product[6]+vec_product[7]+vec_product[8]);
    end // this is not quite right
    endgenerate
    */
    
    TYPE intermediate [CONV_SIZE-1:0];
    genvar i;
    generate
        assign intermediate[0] = vec_product[0];
        for(i = 1; i < CONV_SIZE; i++) begin
            assign intermediate[i] = intermediate[i-1] + vec_product[i];
        end
    endgenerate
    
    assign res = bias + intermediate[CONV_SIZE-1];
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
    //assign y[31:22] = 10'b0;
    //assign y[21:0] = x[31:10];
    assign y = x > 0 ? x >> 10 : 32'b0;
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


module network2
    (input TYPE img[3-1:0][32-1:0][32-1:0],
     input TYPE weights1[1-1:0][3*3*3-1:0],
     input TYPE bias1[1-1:0],
     input TYPE weights2[2-1:0][3*3*1-1:0],
     input TYPE bias2[2-1:0],
     input TYPE weights3[4-1:0][3*3*2-1:0],
     input TYPE bias3[4-1:0],
     input TYPE weights4[2-1:0][1*1*4-1:0],
     input TYPE bias4[2-1:0],
     input TYPE weights5[4-1:0][3*3*2-1:0],
     input TYPE bias5[4-1:0],
     input TYPE weights6[2-1:0][1*1*4-1:0],
     input TYPE bias6[2-1:0],
     input TYPE weights7[4-1:0][3*3*2-1:0],
     input TYPE bias7[4-1:0],
     input TYPE weights8[2-1:0][1*1*4-1:0],
     input TYPE bias8[2-1:0],
     input TYPE weights9[1-1:0][1*1*2-1:0],
     input TYPE bias9[1-1:0],
     output TYPE pred[1-1:0]);
    
    TYPE res1[1-1:0][32-1:0][32-1:0];
    run_channels #(32, 32, 3, 1, 3, 1) r1(img, weights1, bias1, res1);
    
    TYPE res1_pooled[1-1:0][16 - 1:0][16 - 1:0];
    pool3d #(32, 32, 1) p1(res1, res1_pooled);
    
    TYPE res2[2-1:0][16-1:0][16-1:0];
    run_channels #(16, 16, 3, 1, 1, 2) r2(res1_pooled, weights2, bias2, res2);
    
    TYPE res2_pooled[2-1:0][8 - 1:0][8 - 1:0];
    pool3d #(16, 16, 2) p2(res2, res2_pooled);
    
    TYPE res3[4-1:0][8 - 1:0][8 - 1:0];
    run_channels #(8, 8, 3, 1, 2, 4) r3(res2_pooled, weights3, bias3, res3);
    
    TYPE res3_pooled[4-1:0][4-1:0][4-1:0];
    pool3d #(8, 8, 4) p3(res3, res3_pooled);
    
    TYPE res4[2-1:0][4 - 1:0][4 - 1:0];
    run_channels #(4, 4, 1, 0, 4, 2) r4(res3_pooled, weights4, bias4, res4);
    
    TYPE res5[4-1:0][4 - 1:0][4 - 1:0];
    run_channels #(4, 4, 3, 1, 2, 4) r5(res4, weights5, bias5, res5);
    
    TYPE res5_pooled[4-1:0][2-1:0][2-1:0];
    pool3d #(4, 4, 4) p5(res5, res5_pooled);
    
    TYPE res6[2-1:0][2 - 1:0][2 - 1:0];
    run_channels #(2, 2, 1, 0, 4, 2) r6(res5_pooled, weights6, bias6, res6);
        
    TYPE res7[4-1:0][2 - 1:0][2 - 1:0];
    run_channels #(2, 2, 3, 1, 2, 4) r7(res6, weights7, bias7, res7);
    
    TYPE res7_pooled[4-1:0][1-1:0][1-1:0];
    pool3d #(2, 2, 4) p7(res7, res7_pooled);
    
    TYPE res8[2-1:0][1 - 1:0][1 - 1:0];
    run_channels #(1, 1, 1, 0, 4, 2) r8(res7_pooled, weights8, bias8, res8);  
    
    TYPE pred_unshaped[1-1:0][0:0][0:0];
    run_channels #(1, 1, 1, 0, 2, 1, 0) r11(res8, weights9, bias9, pred_unshaped);
    
    genvar chan;
    generate 
        for(chan = 0; chan < 1; chan++) begin
            assign pred[chan] = pred_unshaped[chan][0][0];
        end
    endgenerate
    
endmodule: network2

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

module top4 #(parameter outD = 10, parameter N = 1)
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
    
endmodule: top4

module top (input logic clk, 
            input logic switch[7:0],
            output logic led[7:0]);

    genvar i, j, k;
    
    TYPE img[3072-1:0];
    cifar2 I0(.*);
    TYPE img_rs[3-1:0][32-1:0][32-1:0];
    
    generate for (i = 0; i < 3; i = i+1) begin
        for (j = 0; j < 32; j = j+1) begin
            for (k = 0; k < 32; k = k+1) begin
                assign img_rs[i][j][k] = img[32*32*i + 32*j + k];
            end
        end
    end
    
    TYPE weights1[1*3*3*3-1:0];
    TYPE bias1[1-1:0];
    TYPE weights2[2*3*3*1-1:0];
    TYPE bias2[2-1:0];
    TYPE weights3[4*3*3*2-1:0];
    TYPE bias3[4-1:0];
    TYPE weights4[2*1*1*4-1:0];
    TYPE bias4[2-1:0];
    TYPE weights5[4*3*3*2-1:0];
    TYPE bias5[4-1:0];
    TYPE weights6[2*1*1*4-1:0];
    TYPE bias6[2-1:0];
    TYPE weights7[4*3*3*2-1:0];
    TYPE bias7[4-1:0];
    TYPE weights8[2*1*1*4-1:0];
    TYPE bias8[2-1:0];
    TYPE weights9[1*1*1*2-1:0];
    TYPE bias9[1-1:0];
    
    parameters2 p(weights1, bias1, 
                  weights2, bias2, 
                  weights3, bias3, 
                  weights4, bias4, 
                  weights5, bias5, 
                  weights6, bias6, 
                  weights7, bias7, 
                  weights8, bias8, 
                  weights9, bias9);
    
    TYPE weights1_rs[1-1:0][3*3*3-1:0];
    for (i = 0; i < 1; i = i+1) begin
        for (j = 0; j < 3*3*3; j = j+1) begin
            assign weights1_rs[i][j] = weights1[3*3*3*i + j];
        end
    end
    
    TYPE weights2_rs[2-1:0][3*3*1-1:0];
    for (i = 0; i < 2; i = i+1) begin
        for (j = 0; j < 3*3*1; j = j+1) begin
            assign weights2_rs[i][j] = weights2[3*3*1*i + j];
        end
    end
    
    TYPE weights3_rs[4-1:0][3*3*2-1:0];
    for (i = 0; i < 4; i = i+1) begin
        for (j = 0; j < 3*3*2; j = j+1) begin
            assign weights3_rs[i][j] = weights3[3*3*2*i + j];
        end
    end
    
    TYPE weights4_rs[2-1:0][1*1*4-1:0];
    for (i = 0; i < 2; i = i+1) begin
        for (j = 0; j < 1*1*4; j = j+1) begin
            assign weights4_rs[i][j] = weights4[1*1*4*i + j];
        end
    end
        
    TYPE weights5_rs[4-1:0][3*3*2-1:0];
    for (i = 0; i < 4; i = i+1) begin
        for (j = 0; j < 3*3*2; j = j+1) begin
            assign weights5_rs[i][j] = weights5[3*3*2*i + j];
        end
    end
        
    TYPE weights6_rs[2-1:0][1*1*4-1:0];
    for (i = 0; i < 2; i = i+1) begin
        for (j = 0; j < 1*1*4; j = j+1) begin
            assign weights6_rs[i][j] = weights6[1*1*4*i + j];
        end
    end
        
    TYPE weights7_rs[4-1:0][3*3*2-1:0];
    for (i = 0; i < 4; i = i+1) begin
        for (j = 0; j < 3*3*2; j = j+1) begin
            assign weights7_rs[i][j] = weights7[3*3*2*i + j];
        end
    end
        
    TYPE weights8_rs[2-1:0][1*1*4-1:0];
    for (i = 0; i < 2; i = i+1) begin
        for (j = 0; j < 1*1*4; j = j+1) begin
            assign weights8_rs[i][j] = weights8[1*1*4*i + j];
        end
    end
        
    TYPE weights9_rs[1-1:0][1*1*2-1:0];
    for (i = 0; i < 1; i = i+1) begin
        for (j = 0; j < 1*1*2; j = j+1) begin
            assign weights9_rs[i][j] = weights9[1*1*2*i + j];
        end
    end 
    endgenerate
    
    TYPE bias2_[1:0];
    TYPE bias4_[1:0];
    TYPE bias6_[1:0];
    TYPE bias8_[1:0];
    
    TYPE pred[0:0];
    network2 n(img_rs, weights1_rs, bias1, 
               weights2_rs, bias2_, 
               weights3_rs, bias3, 
               weights4_rs, bias4_, 
               weights5_rs, bias5, 
               weights6_rs, bias6_, 
               weights7_rs, bias7, 
               weights8_rs, bias8_, 
               weights9_rs, bias9, pred);

    assign bias2_[0] = switch[0];
    assign bias2_[1] = switch[1];
    assign bias4_[0] = switch[2];
    assign bias4_[1] = switch[3];
    assign bias6_[0] = switch[4];
    assign bias6_[1] = switch[5];
    assign bias8_[0] = switch[6];
    assign bias8_[1] = switch[7];
    
    always_ff @(posedge clk) begin
        led[0] <= pred[0];
    end

endmodule: top

module top8 (input logic clk, 
            input logic switch[7:0],
            output logic led[7:0]);

    logic thing1[1000-1:0];
    logic thing2[1000-1:0];

    always_ff @(posedge clk, negedge clk) begin
        if (clk) begin
            thing1[0] <= switch[6];
            thing1[999:1] <= thing2[998:0];
        end else begin
            thing1[0] <= switch[1];
            thing2[999:1] <= thing1[998:0];
            led <= thing2[999-:8];
        end
    end
    
endmodule: top8

module top6 (input logic clk, output logic test);
    //do something computationally intensive
    logic stuff[65536-1:0];
    logic sum[65536-1:0];
    assign stuff[0] = clk;
    assign sum[0] = 0;
    genvar i;
    generate
        for(i = 1; i < 65536; i++) begin
            assign stuff[i] = ~stuff[i-1];
            assign sum[i] = sum[i-1]+stuff[i];
        end
    endgenerate
    
    assign test = sum[65536-1];
    
endmodule

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

module add2d #(parameter H = 8, parameter W = 8)
    (input TYPE a[H-1:0][W-1:0],
     input TYPE b[H-1:0][W-1:0],
     output TYPE c[H-1:0][W-1:0]);
     
    genvar i,j;
    generate
        for(i = 0; i < H; i++) begin
            for(j = 0; j < W; j++) begin
                assign c[i][j] = a[i][j]+b[i][j];
            end
        end
    endgenerate 
endmodule

module setzero2d #(parameter H = 8, parameter W = 8)
    (input logic reset,
     output TYPE arr[H-1:0][W-1:0]);
     
    genvar i, j;
    generate
        for(i = 0; i < H; i++) begin
            for(j = 0; j < W; j++) begin
                assign arr[i][j] = 0;
//                always_ff @(posedge reset) begin
//                    arr[i][j] <= 0;
//                end
            end
        end
    endgenerate 
endmodule 

module pool_fn_opt #(parameter N = 1)
    (input TYPE vals[4*N-1:0],
     output TYPE res[N-1:0]);
     
    genvar i;
    generate for (i = 0; i < N; i++) begin
        pool_fn p(vals[4*i], vals[4*i+1], vals[4*i+2], vals[4*i+3], res[i]);
    end endgenerate
endmodule: pool_fn_opt

module test_pool_fn_opt #(parameter N = 4) ();

    TYPE vals[4*N-1:0];
    TYPE res[N-1:0];
    genvar i;
    generate for (i = 0; i < 4*N; i++) begin
        assign vals[i] = i;
    end endgenerate
    
    pool_fn_opt #(N) p(.*);

endmodule: test_pool_fn_opt

module conv_block #(parameter N = 1, S = 3)
    (input TYPE params[S*S*N-1:0],
     input TYPE weights[S*S-1:0],
     input TYPE bias,
     output TYPE res[N-1:0]);
     
    genvar i;
    generate for (i = 0; i < N; i++) begin
        conv_ind #(S*S) c(params[S*S*(i+1)-1 -: S*S], weights, bias, res[i]);
    end endgenerate
    
endmodule

module test_conv_block ();
    TYPE params[3*3*4-1:0];
    TYPE weights[3*3-1:0];
    TYPE bias = 2;
    TYPE res[4-1:0];
    
    genvar i;
    generate for (i = 0; i < 3*3*4; i++) begin
        assign params[i] = i;
    end
    
    for (i = 0; i < 9; i++) begin
        assign weights[i] = 1;
    end endgenerate
    
    conv_block #(4, 3) cb(.*);
endmodule: test_conv_block

module conv_siso #(parameter H = 8, parameter W = 8, parameter N = 4, S = 3)
    (input logic clk,
     input logic reset,
     input TYPE arr[H-1:0][W-1:0],
     input TYPE weights[S*S-1:0],
     input TYPE bias,
     input TYPE conv_block_out[N-1:0],
     output TYPE conv_block_in[S*S*N-1:0],
     output TYPE conv_block_weights[S*S-1:0],
     output TYPE conv_block_bias,
     output TYPE out[H-1:0][W-1:0],
     output logic done); 
     
     //put padding logic into this module
     int i = 0;
     assign done = (i >= H*W + N);
     assign conv_block_weights = weights;
     assign conv_block_bias = bias;
     
     integer j, row, col, old_row, old_col;
     always_ff @(posedge clk) begin
        if (reset) begin
            i = 0;
        end else if (~done) begin
            for (j = 0; j < N; j++) begin
                row = (i+j)/W;
                col = (i+j)%W;
                
                old_row = (i-N+j)/W;
                old_col = (i-N+j)%W;
                out[old_row][old_col] = conv_block_out[j];
                
                conv_block_in[S*S*j] = row-1 < 0 | col-1 < 0 | row-1 >= H | col-1 >= W ? 0 : arr[row-1][col-1];
                conv_block_in[S*S*j+1] = row-1 < 0 | col < 0 | row-1 >= H | col >= W ? 0 : arr[row-1][col];
                conv_block_in[S*S*j+2] = row-1 < 0 | col+1 < 0 | row-1 >= H | col+1 >= W ? 0 : arr[row-1][col+1];
                conv_block_in[S*S*j+3] = row < 0 | col-1 < 0 | row >= H | col-1 >= W ? 0 : arr[row][col-1];
                conv_block_in[S*S*j+4] = row < 0 | col < 0 | row >= H | col >= W ? 0 : arr[row][col];
                conv_block_in[S*S*j+5] = row < 0 | col+1 < 0 | row >= H | col+1 >= W ? 0 : arr[row][col+1];
                conv_block_in[S*S*j+6] = row+1 < 0 | col-1 < 0 | row+1 >= H | col-1 >= W ? 0 : arr[row+1][col-1];
                conv_block_in[S*S*j+7] = row+1 < 0 | col < 0 | row+1 >= H | col >= W ? 0 : arr[row+1][col];
                conv_block_in[S*S*j+8] = row+1 < 0 | col+1 < 0 | row+1 >= H | col+1 >= W ? 0 : arr[row+1][col+1];
            end
            
            i = i+N;
        end
     end  
endmodule: conv_siso

module test_conv_siso ();
    TYPE arr[8-1:0][8-1:0];
    TYPE out[8-1:0][8-1:0];
    TYPE weights[3*3-1:0];
    TYPE bias;
    
    genvar i, j;
    generate for (i = 0; i < 8; i++) begin
        for (j = 0; j < 8; j++) begin
            assign arr[i][j] = 8*i+j;
        end
    end
    
    for (i = 0; i < 9; i++) begin
        assign weights[i] = 1;
    end endgenerate;
    
    assign bias = 2;
    
    TYPE conv_block_in[3*3*1-1:0];
    TYPE conv_block_out[1-1:0];
    TYPE conv_block_weights[3*3-1:0];
    TYPE conv_block_bias;
    conv_block #(1, 3) cb(conv_block_in, conv_block_weights, conv_block_bias, conv_block_out);
    
    logic done, clk, reset;
    initial begin
        clk = 1;
        reset = 1;
        #10
        reset = 0;
        forever #10 clk = ~clk;
    end
    
    conv_siso #(8, 8, 1, 3) cs(.*);
endmodule: test_conv_siso

module conv_miso #(parameter H = 8, parameter W = 8, parameter D = 3, parameter N = 4, parameter S = 3)
    (input logic clk,
     input logic  reset,
     input        TYPE arr[D-1:0][H-1:0][W-1:0],
     input        TYPE weights[D-1:0][S*S-1:0],
     input        TYPE bias,
     input TYPE conv_block_out[N-1:0],
     output TYPE conv_block_in[S*S*N-1:0],
     output TYPE conv_block_weights[S*S-1:0],
     output TYPE conv_block_bias,
     output       TYPE out[H-1:0][W-1:0],
     output logic done);

    int i = 0;
    assign done = (i == D-1);
    
    TYPE siso_arr[H-1:0][W-1:0];
    TYPE siso_weights[S*S-1:0];

    TYPE siso_out[H-1:0][W-1:0];
    TYPE tmp[H-1:0][W-1:0];
    add2d #(H,W) A(siso_out, out, tmp);
    
    logic siso_done, siso_reset;  
    conv_siso #(H, W, N, S) SISO(clk, siso_reset, siso_arr, siso_weights, bias, 
                                 conv_block_out, conv_block_in, conv_block_weights, conv_block_bias,
                                 siso_out, siso_done);
    assign siso_reset = reset | siso_done;
    TYPE out_tmp[H-1:0][W-1:0];
    setzero2d #(H, W) SZ(reset, out_tmp);

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            out <= out_tmp;         
            i <= 0;
        end else if (~done & siso_done) begin
            out <= siso_out;
            siso_weights <= weights[i];
            siso_arr <= arr[i];
            i <= i+1;
        end
    end
    
endmodule

module test_conv_miso();
    TYPE arr[3-1:0][8-1:0][8-1:0];
    TYPE out[8-1:0][8-1:0];
    TYPE weights[3-1:0][3*3-1:0];
    TYPE bias;
    
    genvar i, j, k;
    generate for (k = 0; k < 3; k++) begin
        for (i = 0; i < 8; i++) begin
            for (j = 0; j < 8; j++) begin
                assign arr[k][i][j] = 64*k + 8*i+j;
            end
        end
    end
    
    for (j = 0; j < 3; j++) begin
        for (i = 0; i < 9; i++) begin
            assign weights[j][i] = 1;
        end
    end endgenerate;
    
    assign bias = 2;
    
    TYPE conv_block_in[3*3*1-1:0];
    TYPE conv_block_out[1-1:0];
    TYPE conv_block_weights[3*3-1:0];
    TYPE conv_block_bias;
    conv_block #(1, 3) cb(conv_block_in, conv_block_weights, conv_block_bias, conv_block_out);
    
    logic done, clk, reset;
    initial begin
        clk = 1;
        reset = 1;
        #10
        reset = 0;
        forever #10 clk = ~clk;
    end
    
    conv_miso #(8, 8, 3, 1, 3) cs(.*);

endmodule: test_conv_miso

//H, W: height and width
//D, K: in and out dim
//N: number of conv units
//S: kernel size
module conv_mimo #(parameter H = 8, parameter W = 8, parameter D = 3, parameter K = 3, parameter N = 4, S = 3)
    (input logic clk,
     input logic  reset,
     input        TYPE arr[D-1:0][H-1:0][W-1:0],
     input        TYPE weights[K-1:0][D-1:0][S*S-1:0],
     input        TYPE biases[K-1:0],
     input TYPE conv_block_out[N-1:0],
     output TYPE conv_block_in[S*S*N-1:0],
     output TYPE conv_block_weights[S*S-1:0],
     output TYPE conv_block_bias,
     output       TYPE out[K-1:0][H-1:0][W-1:0],
     output logic done);
     
    int i = 0;
    assign done = (i == K);
    
    logic miso_reset, miso_done;
    TYPE miso_weights[D-1:0][S*S-1:0];
    TYPE miso_bias;
    TYPE miso_out[H-1:0][W-1:0];
    conv_miso #(H, W, D, N, S) MISO(clk, miso_reset, arr, miso_weights, miso_bias, 
                                    conv_block_out, conv_block_in, conv_block_weights, conv_block_bias,
                                    miso_out, miso_done);
   assign miso_reset = reset | miso_done;
    
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            i <= 0;
        end else if (~done & miso_done) begin
            out[i] <= miso_out;
            miso_weights <= weights[i];
            miso_bias <= biases[i];
            i <= i+1;
        end
    end
endmodule

module test_conv_mimo();
    TYPE arr[3-1:0][8-1:0][8-1:0];
    TYPE out[3-1:0][8-1:0][8-1:0];
    TYPE weights[3-1:0][3-1:0][3*3-1:0];
    TYPE biases[3-1:0];
    
    genvar i, j, k;
    generate for (k = 0; k < 3; k++) begin
        for (i = 0; i < 8; i++) begin
            for (j = 0; j < 8; j++) begin
                assign arr[k][i][j] = 64*k + 8*i+j;
            end
        end
    end
    
    for (k = 0; k < 3; k++) begin
        for (j = 0; j < 3; j++) begin
            for (i = 0; i < 9; i++) begin
                assign weights[k][j][i] = 1;
            end
        end
    end endgenerate;
    
    assign biases[0] = 2;
    assign biases[1] = 2;
    assign biases[2] = 2;
    
    TYPE conv_block_in[3*3*1-1:0];
    TYPE conv_block_out[1-1:0];
    TYPE conv_block_weights[3*3-1:0];
    TYPE conv_block_bias;
    conv_block #(1, 3) cb(conv_block_in, conv_block_weights, conv_block_bias, conv_block_out);
    
    logic done, clk, reset;
    initial begin
        clk = 1;
        reset = 1;
        #10
        reset = 0;
        forever #10 clk = ~clk;
    end
    
    conv_mimo #(8, 8, 3, 3, 1, 3) cm(.*);

endmodule: test_conv_mimo

module relu_block #(parameter N = 4)
    (input TYPE inp[N-1:0],
     output TYPE out[N-1:0]);

    genvar i;
    generate for (i = 0; i < N; i++) begin
        relu R(inp[i],out[i]);
    end endgenerate
endmodule


module relu_layer #(parameter H = 8, parameter W = 8, parameter D = 4, parameter N = 4)
    (input clk,
     input reset,
     input TYPE relu_block_out[N-1:0],
     output TYPE relu_block_in[N-1:0],
     input TYPE arr[D-1:0][H-1:0][W-1:0],
     output TYPE out[D-1:0][H-1:0][W-1:0],
     output logic done);
          
    //maybe do this in place?
    int i = 0;
    assign done = (i >= D*W*H + N);
    
    integer j, layer, row, col, old_layer, old_row, old_col;
    
    always_ff @(posedge clk) begin
        if (reset) begin
            i = 0;
        end else if (~done) begin
            for (j = 0; j < N; j++) begin
                layer = (i+j)/(W*H);
                row = ((i+j)%(W*H))/W;
                col = (i+j)%W;
                
                old_layer = (i-N+j)/(W*H);
                old_row = ((i-N+j)%(W*H))/W;
                old_col = (i-N+j)%W;
                out[old_layer][old_row][old_col] = relu_block_out[j];
                
                relu_block_in[j] = arr[layer][row][col];
            end
            
            i = i+N;
        end
    end

endmodule

module test_relu_layer ();
    TYPE arr[4-1:0][8-1:0][8-1:0];
    TYPE out[4-1:0][8-1:0][8-1:0];
    
    genvar i, j, k;
    generate for (k = 0; k < 4; k++) begin
      for (i = 0; i < 8; i++) begin
        for (j = 0; j < 8; j++) begin
            assign arr[k][i][j] = 1024*(64*k + 8*i+j);
        end
      end
    end endgenerate
    
    TYPE relu_block_in[4-1:0];
    TYPE relu_block_out[4-1:0];
    
    relu_block #(4) rb(relu_block_in, relu_block_out);
    
    logic done, clk, reset;
    initial begin
        clk = 1;
        reset = 1;
        #10
        reset = 0;
        forever #10 clk = ~clk;
    end
    
    relu_layer #(8, 8, 4, 4) rl(.*);
endmodule: test_relu_layer

module up_pool_opt #(parameter H = 8, parameter W = 8, parameter N = 4)
    (input clk,
     input reset,
     input TYPE arr[H-1:0][W-1:0],
     input TYPE pool_fn_res[N-1:0],
     output TYPE pool_fn_vals[4*N-1:0],
     output TYPE pooled[H/2-1:0][W/2-1:0],
     output logic done);
     
     int i = 0;
     assign done = (i >= H*W/4 + N);
     
     integer j, row, col, old_row, old_col;
     logic in = 1;
     
     always_ff @(posedge clk) begin
        if (reset) begin
            i = 0;
            in = 1;
        end else if (~done) begin
            for (j = 0; j < N; j++) begin
                if (~in) begin
                    old_row = (i-N+j)/(W/2);
                    old_col = (i-N+j)%(W/2);
                    pooled[old_row][old_col] = pool_fn_res[j];
                end else begin
                    row = (i+j)/(W/2);
                    col = (i+j)%(W/2);
                    pool_fn_vals[4*j] = arr[2*row][2*col];
                    pool_fn_vals[4*j+1] = arr[2*row][2*col + 1];
                    pool_fn_vals[4*j+2] = arr[2*row + 1][2*col];
                    pool_fn_vals[4*j+3] = arr[2*row + 1][2*col + 1];
                end
            end
            
            in = ~in;
            i = i+N;
        end
     end
endmodule: up_pool_opt

module test_up_pool_opt #(parameter H = 8, parameter W = 8, parameter N = 16) ();
    TYPE pool_fn_res[N-1:0];
    TYPE pool_fn_vals[4*N-1:0];
    pool_fn_opt #(N) p(pool_fn_vals, pool_fn_res);

    TYPE arr[H-1:0][W-1:0];
    genvar i, j;
    generate for (i = 0; i < H; i++) begin
        for (j = 0; j < W; j++) begin
            assign arr[i][j] = W*i + j;
        end
    end endgenerate
    
    logic clk;
    always begin
        #5 clk = 1;
        #5 clk = 0;
    end
    
    logic reset = 0;
    logic done = 0;
    
    TYPE pooled[H/2-1:0][W/2-1:0];
    up_pool_opt #(H, W, N) po(clk, reset, arr, pool_fn_res, pool_fn_vals, pooled, done);
    
endmodule: test_up_pool_opt

module pool_opt #(parameter H = 8, parameter W = 8, parameter N = 4)
    (input clk,
     input reset,
     input TYPE arr[H-1:0][W-1:0],
     input TYPE pool_fn_res[N-1:0],
     output TYPE pool_fn_vals[4*N-1:0],
     output TYPE pooled[H/2-1:0][W/2-1:0],
     output logic done);
     
     int i = 0;
     assign done = (i >= H*W/4 + N);
     
     integer j, row, col, old_row, old_col;
     
     always_ff @(posedge clk) begin
        if (reset) begin
            i = 0;
        end else if (~done) begin
            for (j = 0; j < N; j++) begin
                row = (i+j)/(W/2);
                col = (i+j)%(W/2);
                
                old_row = (i-N+j)/(W/2);
                old_col = (i-N+j)%(W/2);
                pooled[old_row][old_col] = pool_fn_res[j];
                
                pool_fn_vals[4*j] = arr[2*row][2*col];
                pool_fn_vals[4*j+1] = arr[2*row][2*col + 1];
                pool_fn_vals[4*j+2] = arr[2*row + 1][2*col];
                pool_fn_vals[4*j+3] = arr[2*row + 1][2*col + 1];
            end
            
            i = i+N;
        end
     end
endmodule: pool_opt

module test_pool_opt #(parameter H = 8, parameter W = 8, parameter N = 16) ();
    TYPE pool_fn_res[N-1:0];
    TYPE pool_fn_vals[4*N-1:0];
    pool_fn_opt #(N) p(pool_fn_vals, pool_fn_res);

    TYPE arr[H-1:0][W-1:0];
    genvar i, j;
    generate for (i = 0; i < H; i++) begin
        for (j = 0; j < W; j++) begin
            assign arr[i][j] = W*i + j;
        end
    end endgenerate
    
    logic clk;
    always begin
        #5 clk = 1;
        #5 clk = 0;
    end
    
    logic reset = 0;
    logic done = 0;
    
    TYPE pooled[H/2-1:0][W/2-1:0];
    pool_opt #(H, W, N) po(clk, reset, arr, pool_fn_res, pool_fn_vals, pooled, done);
    
endmodule: test_pool_opt

module pool_layers_opt #(parameter H = 8, parameter W = 8, parameter L = 2, parameter N = 4)
    (input logic clk,
     input logic reset,
     input TYPE arr[L-1:0][H-1:0][W-1:0],
     output TYPE res[L-1:0][H/2-1:0][W/2-1:0],
     output logic done);
     
    TYPE pool_fn_res[N-1:0];
    TYPE pool_fn_vals[4*N-1:0];
    pool_fn_opt #(N) pf(pool_fn_vals, pool_fn_res);
    
    int i;
    assign done = (i == L+1);
    
    logic pool_reset;
    logic pool_done;
    
    TYPE arr_layer[H-1:0][W-1:0];
    TYPE pooled_layer[H/2-1:0][W/2-1:0];
    pool_opt #(H, W, N) p(clk, pool_reset, arr_layer, pool_fn_res, pool_fn_vals, pooled_layer, pool_done);
    assign pool_reset = reset | pool_done;
    
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            i <= 0;
        end else if (~done & pool_done) begin
            res[i-1] <= pooled_layer;
            arr_layer <= arr[i];
            
            i <= i+1;
        end
     end
endmodule: pool_layers_opt
 
module test_pool_layers_opt ();
    TYPE arr[2-1:0][8-1:0][8-1:0];
    TYPE res[2-1:0][4-1:0][4-1:0];
    
    genvar i, j, k;
    generate for (i = 0; i < 2; i++) begin
        for (j = 0; j < 8; j++) begin
            for (k = 0; k < 8; k++) begin
                assign arr[i][j][k] = 64*i + 8*j + k;
            end
        end
    end endgenerate
    
    logic clk, reset;
    initial begin
        clk = 1;
        reset = 1;
        #10
        reset = 0;
        forever #10 clk = ~clk;
    end
    logic done;
    
    pool_layers_opt #(8, 8, 2, 4) p(.*);
    
endmodule: test_pool_layers_opt

module network_opt #(parameter H = 32, 
                     parameter W = 32, 
                     parameter N = 1024)
    (input TYPE img[3-1:0][H-1:0][W-1:0],
     input TYPE weights1[8-1:0][3-1:0][3*3-1:0],
     input TYPE bias1[8-1:0],
     input TYPE weights2[16-1:0][3-1:0][3*8-1:0],
     input TYPE bias2[16-1:0],
     input TYPE weights3[32-1:0][3-1:0][3*16-1:0],
     input TYPE bias3[32-1:0],
     input TYPE weights4[16-1:0][1-1:0][1*32-1:0],
     input TYPE bias4[16-1:0],
     input TYPE weights5[32-1:0][3-1:0][3*16-1:0],
     input TYPE bias5[32-1:0],
     input TYPE weights6[16-1:0][1-1:0][1*32-1:0],
     input TYPE bias6[16-1:0],
     input TYPE weights7[32-1:0][3-1:0][3*16-1:0],
     input TYPE bias7[32-1:0],
     input TYPE weights8[16-1:0][1-1:0][1*32-1:0],
     input TYPE bias8[16-1:0],
     input TYPE weights9[10-1:0][1-1:0][1*16-1:0],
     input TYPE bias9[10-1:0],
     input logic clk,
     input logic reset_,
     output logic finished,
     output TYPE pred[10-1:0]); 
    
    //initialize memory
    TYPE convlayer1[8-1:0][H-1:0][W-1:0];
    TYPE relulayer1[8-1:0][H-1:0][W-1:0];
    TYPE poollayer1[8-1:0][H/2 - 1:0][W/2 - 1:0];    
    TYPE convlayer2[16-1:0][H/2-1:0][W/2-1:0];
    TYPE relulayer2[16-1:0][H/2-1:0][W/2-1:0];
    TYPE poollayer2[16-1:0][H/4 - 1:0][W/4 - 1:0];
    TYPE convlayer3[32-1:0][H/4 - 1:0][W/4 - 1:0];
    TYPE relulayer3[32-1:0][H/4 - 1:0][W/4 - 1:0];
    TYPE poollayer3[32-1:0][H/8-1:0][W/8-1:0];
    TYPE convlayer4[16-1:0][H/8 - 1:0][W/8 - 1:0];
    TYPE relulayer4[16-1:0][H/8 - 1:0][W/8 - 1:0];
    TYPE convlayer5[32-1:0][H/8 - 1:0][W/8 - 1:0];
    TYPE relulayer5[32-1:0][H/8 - 1:0][W/8 - 1:0];
    TYPE poollayer5[32-1:0][H/16-1:0][W/16-1:0];
    TYPE convlayer6[16-1:0][H/16 - 1:0][W/16 - 1:0];
    TYPE relulayer6[16-1:0][H/16 - 1:0][W/16 - 1:0];
    TYPE convlayer7[32-1:0][H/16 - 1:0][W/16 - 1:0];
    TYPE relulayer7[32-1:0][H/16 - 1:0][W/16 - 1:0];
    TYPE poollayer7[32-1:0][H/32-1:0][W/32-1:0];
    TYPE convlayer8[16-1:0][H/32 - 1:0][W/32 - 1:0];
    TYPE relulayer8[16-1:0][H/32 - 1:0][W/32 - 1:0];
    TYPE convlayer9[10-1:0][0:0][0:0];
    
    //initialize blocks
    
    TYPE conv3_in[9*1024-1:0];
    TYPE conv3_weights[9-1:0];
    TYPE conv3_bias;
    TYPE conv3_out[1024-1:0];
    conv_block #(1024, 3) C3(conv3_in, conv3_weights, conv3_bias, conv3_out);
    
    TYPE conv1_in[1024-1:0];
    TYPE conv1_weights[0:0];
    TYPE conv1_bias;
    TYPE conv1_out[1024-1:0];
    conv_block #(1024, 1) C1(conv1_in, conv1_weights, conv1_bias, conv1_out);
    
    TYPE relu_in[1024-1:0];
    TYPE relu_out[1024-1:0];
    relu_block #(1024) R(relu_in, relu_out);
    
    TYPE pool_in[4*1024-1:0];
    TYPE pool_out[1024-1:0];
    pool_fn_opt #(1024) P(pool_in, pool_out);
    
    //use this to control the computation
    logic done[21:0];
    logic [21:0] reset;
    
//    module conv_mimo #(parameter H = 8, parameter W = 8, parameter D = 3, parameter K = 3, parameter N = 4, S = 3)
//        (input logic clk,
//         input logic  reset,
//         input        TYPE arr[D-1:0][H-1:0][W-1:0],
//         input        TYPE weights[K-1:0][D-1:0][S*S-1:0],
//         input        TYPE biases[K-1:0],
//         input TYPE conv_block_out[N-1:0],
//         output TYPE conv_block_in[S*S*N-1:0],
//         output TYPE conv_block_weights[S*S-1:0],
//         output TYPE conv_block_bias,
//         output       TYPE out[K-1:0][H-1:0][W-1:0],
//         output logic done);
    int stage = 0;
    
    TYPE conv3_in1[9*1024-1:0];
    TYPE conv3_in2[9*1024-1:0];
    TYPE conv3_in3[9*1024-1:0];
    TYPE conv3_in4[9*1024-1:0];
    TYPE conv3_in5[9*1024-1:0];
    assign conv3_in = (stage == 0) ? conv3_in1 : (stage == 3) ? conv3_in2 : (stage == 6) ? conv3_in3 : (stage == 4) ? conv3_in4 : conv3_in5;
    

    TYPE conv1_in1[1024-1:0];
    TYPE conv1_in2[1024-1:0];
    TYPE conv1_in3[1024-1:0];
    TYPE conv1_in4[1024-1:0];
    assign conv1_in = (stage == 9) ? conv1_in1 : (stage == 14) ? conv1_in2 : (stage == 19) ? conv1_in3 : conv1_in4;
    
    TYPE conv3_weights1[9-1:0];
    TYPE conv3_weights2[9-1:0];
    TYPE conv3_weights3[9-1:0];
    TYPE conv3_weights4[9-1:0];
    TYPE conv3_weights5[9-1:0];
    assign conv3_weights = (stage == 0) ? conv3_weights1 : (stage == 3) ? conv3_weights2 : (stage == 6) ? conv3_weights3 : (stage == 4) ? conv3_weights4 : conv3_weights5;

    TYPE conv1_weights1[1-1:0];
    TYPE conv1_weights2[1-1:0];
    TYPE conv1_weights3[1-1:0];
    TYPE conv1_weights4[1-1:0];
    assign conv1_weights = (stage == 9) ? conv1_weights1 : (stage == 14) ? conv1_weights2 : (stage == 19) ? conv1_weights3 : conv1_weights4;
    
    //initialize conv layers
    conv_mimo #(32, 32, 3, 8, 1024, 3) cm1(.clk(clk), .reset(reset[0]), .arr(img), .weights(weights1), .biases(bias1), .conv_block_out(conv3_out), .conv_block_in(conv3_in1), .conv_block_weights(conv3_weights1), .conv_block_bias(conv3_bias1), .out(convlayer1), .done(done[0]));
    conv_mimo #(16, 16, 8, 16, 1024, 3) cm2(clk, reset[3], poollayer1, weights2, bias2, conv3_out, conv3_in2, conv3_weights2, conv3_bias2, convlayer2, done[3]);
    conv_mimo #(8, 8, 16, 32, 1024, 3) cm3(clk, reset[6], poollayer2, weights3, bias3, conv3_out, conv3_in3, conv3_weights3, conv3_bias3, convlayer3, done[6]);
    conv_mimo #(4, 4, 32, 16, 1024, 1) cm4(clk, reset[9], poollayer3, weights4, bias4, conv1_out, conv1_in1, conv1_weights1, conv1_bias1, convlayer4, done[9]);
    conv_mimo #(4, 4, 16, 32, 1024, 3) cm5(clk, reset[11], relulayer4, weights5, bias5, conv3_out, conv3_in4, conv3_weights4, conv3_bias4, convlayer5, done[11]);
    conv_mimo #(2, 2, 32, 16, 1024, 1) cm6(clk, reset[14], poollayer5, weights6, bias6, conv1_out, conv1_in2, conv1_weights2, conv1_bias2, convlayer6, done[14]);
    conv_mimo #(2, 2, 16, 32, 1024, 3) cm7(clk, reset[16], relulayer6, weights7, bias7, conv3_out, conv3_in5, conv3_weights5, conv3_bias5, convlayer7, done[16]);
    conv_mimo #(1, 1, 32, 16, 1024, 1) cm8(clk, reset[19], poollayer7, weights8, bias8, conv1_out, conv1_in3, conv1_weights3, conv1_bias3, convlayer8, done[19]);
    conv_mimo #(1, 1, 16, 10, 1024, 1) cm9(clk, reset[21], relulayer8, weights9, bias9, conv1_out, conv1_in4, conv1_weights4, conv1_bias4, convlayer9, done[21]);

    //initialize relu layers
    relu_layer #(32, 32, 8, 1024) rl1(clk, reset[1], relu_out, relu_in, convlayer1, relulayer1, done[1]);
    relu_layer #(16, 16, 16, 1024) rl2(clk, reset[4], relu_out, relu_in, convlayer2, relulayer2, done[4]);
    relu_layer #(8, 8, 32, 1024) rl3(clk, reset[7], relu_out, relu_in, convlayer3, relulayer3, done[7]);
    relu_layer #(4, 4, 16, 1024) rl4(clk, reset[10], relu_out, relu_in, convlayer4, relulayer4, done[10]);
    relu_layer #(4, 4, 32, 1024) rl5(clk, reset[12], relu_out, relu_in, convlayer5, relulayer5, done[12]);
    relu_layer #(2, 2, 16, 1024) rl6(clk, reset[15], relu_out, relu_in, convlayer6, relulayer6, done[15]);
    relu_layer #(2, 2, 32, 1024) rl7(clk, reset[17], relu_out, relu_in, convlayer7, relulayer7, done[17]);
    relu_layer #(1, 1, 16, 1024) rl8(clk, reset[20], relu_out, relu_in, convlayer8, relulayer8, done[20]);
    
    //initialize pool layers
    pool_layers_opt #(32, 32, 1024) pl1(clk, reset[2], relulayer1, poollayer1, done[2]);
    pool_layers_opt #(16, 16, 1024) pl2(clk, reset[5], relulayer2, poollayer2, done[5]);
    pool_layers_opt #(8, 8, 1024) pl3(clk, reset[8], relulayer3, poollayer3, done[8]);
    pool_layers_opt #(4, 4, 1024) pl5(clk, reset[13], relulayer5, poollayer5, done[13]);
    pool_layers_opt #(2, 2, 1024) pl7(clk, reset[18], relulayer7, poollayer7, done[18]);
    
    //now run all layers
    logic ready;

    assign ready = done[stage];
    assign finished = (stage >= 21);
    
    always_ff @(posedge clk, posedge reset_) begin
        if (reset_) begin
            stage <= 0;
            reset <= 21'h1FFFF;
        end else if (ready & ~finished) begin
            stage <= stage + 1; //reset that layer
            reset[stage] <= 1;
        end else begin
            stage <= stage;
            reset <= 21'h0;
        end
    end
    
    //just some minor reshaping
    genvar chan;
    generate 
        for(chan = 0; chan < 10; chan++) begin
            assign pred[chan] = convlayer9[chan][0][0];
        end
    endgenerate
    
endmodule: network_opt

module test_network_opt();
    TYPE img[3-1:0][32-1:0][32-1:0];
    TYPE weights1[8-1:0][3-1:0][3*3-1:0];
    TYPE bias1[8-1:0];
    TYPE weights2[16-1:0][3-1:0][3*8-1:0];
    TYPE bias2[16-1:0];
    TYPE weights3[32-1:0][3-1:0][3*16-1:0];
    TYPE bias3[32-1:0];
    TYPE weights4[16-1:0][1-1:0][1*32-1:0];
    TYPE bias4[16-1:0];
    TYPE weights5[32-1:0][3-1:0][3*16-1:0];
    TYPE bias5[32-1:0];
    TYPE weights6[16-1:0][1-1:0][1*32-1:0];
    TYPE bias6[16-1:0];
    TYPE weights7[32-1:0][3-1:0][3*16-1:0];
    TYPE bias7[32-1:0];
    TYPE weights8[16-1:0][1-1:0][1*32-1:0];
    TYPE bias8[16-1:0];
    TYPE weights9[10-1:0][1-1:0][1*16-1:0];
    TYPE bias9[10-1:0];
    logic clk;
    logic reset_;
     
    logic finished;
    TYPE pred[10-1:0];
    
    
    TYPE img_og[3072-1:0];
    cifar #(1024) I(img_og);
    
    genvar i, j, k;
    TYPE img_rs[3-1:0][32-1:0][32-1:0];
    generate for (i = 0; i < 3; i = i+1) begin
        for (j = 0; j < 32; j = j+1) begin
            for (k = 0; k < 32; k = k+1) begin
                assign img[i][j][k] = img_og[32*32*i + 32*j + k];
            end
        end
    end
    
    TYPE weights1_og[1*3*3*3-1:0];
    TYPE bias1_og[1-1:0];
    TYPE weights2_og[2*3*3*1-1:0];
    TYPE bias2_og[2-1:0];
    TYPE weights3_og[4*3*3*2-1:0];
    TYPE bias3_og[4-1:0];
    TYPE weights4_og[2*1*1*4-1:0];
    TYPE bias4_og[2-1:0];
    TYPE weights5_og[4*3*3*2-1:0];
    TYPE bias5_og[4-1:0];
    TYPE weights6_og[2*1*1*4-1:0];
    TYPE bias6_og[2-1:0];
    TYPE weights7_og[4*3*3*2-1:0];
    TYPE bias7_og[4-1:0];
    TYPE weights8_og[2*1*1*4-1:0];
    TYPE bias8_og[2-1:0];
    TYPE weights9_og[1*1*1*2-1:0];
    TYPE bias9_og[1-1:0];
    
    parameters2 p(weights1_og, bias1_og, 
                  weights2_og, bias2_og, 
                  weights3_og, bias3_og, 
                  weights4_og, bias4_og, 
                  weights5_og, bias5_og, 
                  weights6_og, bias6_og, 
                  weights7_og, bias7_og, 
                  weights8_og, bias8_og, 
                  weights9_og, bias9_og);
    
    TYPE weights1_rs[1-1:0][3-1:0][3*3-1:0];
    for (i = 0; i < 1; i = i+1) begin
        for (j = 0; j < 3; j = j+1) begin
            for (k = 0; k < 3*3; k++) begin
                assign weights1[i][j][k] = weights1_og[3*3*3*i + j];
            end
        end
    end
    
    TYPE weights2_rs[2-1:0][3-1:0][3*1-1:0];
    for (i = 0; i < 2; i = i+1) begin
        for (j = 0; j < 3; j = j+1) begin
            for (k = 0; k < 3*1; k++) begin
                assign weights2[i][j][k] = weights2_og[3*3*1*i + j];
            end
        end
    end
    
    TYPE weights3_rs[4-1:0][3-1:0][3*2-1:0];
    for (i = 0; i < 4; i = i+1) begin
        for (j = 0; j < 3; j = j+1) begin
            for (k = 0; k < 3*2; k++) begin
                assign weights3[i][j][k] = weights3_og[3*3*2*i + j];
            end
        end
    end
    
    TYPE weights4_rs[2-1:0][1-1:0][1*4-1:0];
    for (i = 0; i < 2; i = i+1) begin
        for (j = 0; j < 1; j = j+1) begin
            for (k = 0; k < 1*4; k++) begin
                assign weights4[i][j][k] = weights4_og[1*1*4*i + j];
            end
        end
    end
        
    TYPE weights5_rs[4-1:0][3-1:0][3*2-1:0];
    for (i = 0; i < 4; i = i+1) begin
        for (j = 0; j < 3; j = j+1) begin
            for (k = 0; k < 3*2; k++) begin
                assign weights5[i][j][k] = weights5_og[3*3*2*i + j];
            end
        end
    end
        
    TYPE weights6_rs[2-1:0][1-1:0][1*4-1:0];
    for (i = 0; i < 2; i = i+1) begin
        for (j = 0; j < 1; j = j+1) begin
            for(k = 0; k < 1*4; k = k+1) begin
                assign weights6[i][j][k] = weights6_og[1*4*i + j];
            end
        end
    end
        
    TYPE weights7_rs[4-1:0][3-1:0][3*2-1:0];
    for (i = 0; i < 4; i = i+1) begin
        for (j = 0; j < 3; j = j+1) begin
            for (k = 0; k < 3*2; k++) begin
                assign weights7[i][j][k] = weights7_og[3*3*2*i + j];
            end
        end
    end
        
    TYPE weights8_rs[2-1:0][1-1:0][1*4-1:0];
    for (i = 0; i < 2; i = i+1) begin
        for (j = 0; j < 1; j = j+1) begin
            for(k = 0; k < 1*4; k+=1) begin
                assign weights8[i][j][k] = weights8_og[1*4*i + j];
            end
        end
    end
        
    TYPE weights9_rs[1-1:0][1-1:0][1*2-1:0];
    for (i = 0; i < 1; i = i+1) begin
        for (j = 0; j < 1; j = j+1) begin
            for (k = 0; k < 1*2; k++) begin
                assign weights9[i][j][k] = weights9_og[1*1*2*i + j];
            end
        end
    end endgenerate
    
    initial begin
        reset_ = 1;
        clk = 1;
        #10
        reset_ = 0;
        forever clk = ~clk;
    end

    network_opt #(32, 32, 1024) no(.*);

endmodule: test_network_opt

module top_opt #(parameter outD = 10, parameter N = 0)
    (output TYPE pred[outD-1:0]);
    
    TYPE img[3072-1:0];
    cifar #(N) I(img);
    
    genvar i, j, k;
    TYPE img_rs[3-1:0][32-1:0][32-1:0];
    generate for (i = 0; i < 3; i = i+1) begin
        for (j = 0; j < 32; j = j+1) begin
            for (k = 0; k < 32; k = k+1) begin
                assign img_rs[i][j][k] = img[32*32*i + 32*j + k];
            end
        end
    end endgenerate
    
    // load data from RAM
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
    
    TYPE weights1_rs[8-1:0][3*3*3-1:0];
    generate for (i = 0; i < 8; i = i+1) begin
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
    
    network_opt #(32, 32, 3, 10) n(img_rs, weights1_rs, bias1, weights2_rs, bias2, weights3_rs, bias3, weights4_rs, bias4, weights5_rs, bias5, weights6_rs, bias6, weights7_rs, bias7, weights8_rs, bias8, weights9_rs, bias9, pred);
    
endmodule: top_opt

module top3();
    
    logic a;
    logic b;
    logic clk;
    logic output_;
    logic input_;
    
    assign b = ~a; //input a, output b
    
    always_ff @(posedge clk) begin
        a <= input_;
        output_ <= b;
    end
    
    initial begin
        clk = 0;
        input_ = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        #10
        @(posedge clk);
        input_ = 1;
        #50
        @(posedge clk);
        input_ = 0;
        #10 $finish;
    end
    
endmodule

/*
module use_y
    (input logic doit,
     input TYPE y[8-1:0][4-1:0][32-1:0][32-1:0],
     output TYPE z[8-1:0][4-1:0][32-1:0][32-1:0]);
     
    genvar i, j, k, l;
    generate
        for(i = 0; i < 8; i++) begin
            for(j = 0; j < 4; j++) begin
                for(k = 0; k < 32; k++) begin
                    for(l = 0; l < 32; l++) begin
                        always_ff @(posedge doit) begin
                            z[i][j][k][l] <= y[i][j][k][l] + 1;
                        end
                    end
                end
            end
        end
    endgenerate 
endmodule

module use_z
    (input logic doit,
     input TYPE z[32-1:0][32-1:0][8-1:0][4-1:0],
     output TYPE y[32-1:0][32-1:0][8-1:0][4-1:0]);
     
    genvar i, j, k, l;
    generate
        for(i = 0; i < 32; i++) begin
            for(j = 0; j < 32; j++) begin
                for(k = 0; k < 8; k++) begin
                    for(l = 0; l < 4; l++) begin
                        always_ff @(posedge doit) begin
                            y[i][j][k][l] <= z[i][j][k][l] - 1;
                        end
                    end
                end
            end
        end
    endgenerate 
endmodule

module sz1d
    (input logic doit,
     output TYPE arr[32768-1:0]);
    
    genvar i;
    generate
        for(i = 0; i < 32768; i++) begin
            always_ff @(posedge doit) begin
                arr[i] <= 0;
            end
        end
    endgenerate 
endmodule

module reshape_1
    (output TYPE x[32-1:0][32-1:0][8-1:0][4-1:0],
     input TYPE x_[32768-1:0]);
     
    genvar i;
     generate
         for(i = 0; i < 32768; i++) begin
            assign x[(i>>10)%32][(i>>5)%32][(i>>2)%8][i%4] = x_[i];
         end
     endgenerate 

endmodule

module reshape_2
    (output TYPE x[8-1:0][4-1:0][32-1:0][32-1:0],
     input TYPE x_[32768-1:0]);
     
    genvar i;
     generate
         for(i = 0; i < 32768; i++) begin
            assign x[(i>>12)%8][(i>>10)%4][(i>>5)%32][i%32] = x_[i];
         end
     endgenerate 

endmodule

module reshape_test();
    
    TYPE y[32768-1:0];
    TYPE z[32768-1:0];
    
    TYPE yr1[32-1:0][32-1:0][8-1:0][4-1:0];
    TYPE yr2[8-1:0][4-1:0][32-1:0][32-1:0];
    reshape_1 ry1(yr1, y);
    reshape_2 ry2(yr2, y);
    
    TYPE zr1[32-1:0][32-1:0][8-1:0][4-1:0];
    TYPE zr2[8-1:0][4-1:0][32-1:0][32-1:0];
    reshape_1 rz1(zr1, z);
    reshape_2 rz2(zr2, z);
    
    logic clz;
    logic cly;   
    logic doz;
    logic doy;
    
    sz1d CZ(clz, z);
    use_y UY(doy, yr2, zr2);
    use_z UZ(doz, yr1, zr1);
    
    initial begin
        doz = 0;
        doy = 0;
        clz = 1;
        cly = 1;
        #10
        clz = 0;
        cly = 0;
        #10
        doz = 1;
        doy = 0;
        #10
        doz = 0;
        doy = 1;
        #10
        doz = 1;
        doy = 0;
        #10
        doz = 0;
        doy = 1;
        #10
        doz = 1;
        doy = 0;
        #10
        doz = 0;
        doy = 1;
        #10
        doz = 1;
        doy = 0;
        #10
        doz = 0;
        doy = 1;
        #10
        $finish;
    end
endmodule
*/


module threebythree(input logic asdf[2:0][2:0],
                    output logic qwer);
    assign qwer = asdf[0][0] + asdf[2][2];
endmodule

module ninebyone();
    logic nbo[8:0];
    logic out;
    threebythree T(nbo, out);
        
    initial begin
        nbo = {1, 0, 0, 0, 1, 0, 1, 1, 1};
        #10
        $finish;
    end

endmodule

module drive_a1 (output logic a, input logic clk, input logic go);
    always_ff @(posedge clk) begin
        if (go) begin
            a <= 1;
        end
    end   
endmodule

module drive_a0 (output logic a, input logic clk, input logic go);
    always_ff @(posedge clk) begin
        if (go) begin
            a <= 0;
        end
    end   
endmodule

module md();
    logic a;
    logic clk;
    logic go0;
    logic go1;  
    
    drive_a1 D1(a, clk, go1);
    drive_a0 D0(a, clk, go0);
    
    initial begin
        clk = 0;
        a = 0;
        go0 = 0;
        go1 = 0;
        #10
        forever #10 clk = ~clk;
    end
    
    initial begin
        #20
        go0 = 1;
        #10
        go0 = 0;
        go1 = 1;
        #20
        go1 = 0;
        go0 = 1;
        #50
        go1 = 0;
        go0 = 0;
    end
endmodule: md
