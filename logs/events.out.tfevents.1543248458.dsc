       �K"	  ����Abrain.Event:24vрy>     ����	m�����A"��
�
Encoder/real_inPlaceholder*
dtype0*/
_output_shapes
:���������*$
shape:���������
f
Encoder/Reshape/shapeConst*
valueB"����  *
dtype0*
_output_shapes
:
�
Encoder/ReshapeReshapeEncoder/real_inEncoder/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
�
KEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *HY��*
dtype0*
_output_shapes
: 
�
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *HY�=*
dtype0*
_output_shapes
: 
�
SEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformKEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
seed2 
�
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
_output_shapes
: *
T0
�
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulSEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
EEncoder/first_layer/fully_connected/kernel/Initializer/random_uniformAddIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0
�
*Encoder/first_layer/fully_connected/kernel
VariableV2*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
1Encoder/first_layer/fully_connected/kernel/AssignAssign*Encoder/first_layer/fully_connected/kernelEEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
/Encoder/first_layer/fully_connected/kernel/readIdentity*Encoder/first_layer/fully_connected/kernel*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
:Encoder/first_layer/fully_connected/bias/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
(Encoder/first_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
	container *
shape:�
�
/Encoder/first_layer/fully_connected/bias/AssignAssign(Encoder/first_layer/fully_connected/bias:Encoder/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
-Encoder/first_layer/fully_connected/bias/readIdentity(Encoder/first_layer/fully_connected/bias*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:�
�
*Encoder/first_layer/fully_connected/MatMulMatMulEncoder/Reshape/Encoder/first_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
+Encoder/first_layer/fully_connected/BiasAddBiasAdd*Encoder/first_layer/fully_connected/MatMul-Encoder/first_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
i
$Encoder/first_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
"Encoder/first_layer/leaky_relu/mulMul$Encoder/first_layer/leaky_relu/alpha+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Encoder/first_layer/leaky_reluMaximum"Encoder/first_layer/leaky_relu/mul+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
LEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *qĜ�
�
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *qĜ=*
dtype0*
_output_shapes
: 
�
TEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
seed2 
�
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
_output_shapes
: 
�
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
FEncoder/second_layer/fully_connected/kernel/Initializer/random_uniformAddJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
+Encoder/second_layer/fully_connected/kernel
VariableV2*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
2Encoder/second_layer/fully_connected/kernel/AssignAssign+Encoder/second_layer/fully_connected/kernelFEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
0Encoder/second_layer/fully_connected/kernel/readIdentity+Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
�
;Encoder/second_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB�*    
�
)Encoder/second_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container *
shape:�
�
0Encoder/second_layer/fully_connected/bias/AssignAssign)Encoder/second_layer/fully_connected/bias;Encoder/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
.Encoder/second_layer/fully_connected/bias/readIdentity)Encoder/second_layer/fully_connected/bias*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
_output_shapes	
:�*
T0
�
+Encoder/second_layer/fully_connected/MatMulMatMulEncoder/first_layer/leaky_relu0Encoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
,Encoder/second_layer/fully_connected/BiasAddBiasAdd+Encoder/second_layer/fully_connected/MatMul.Encoder/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
?Encoder/second_layer/batch_normalization/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:�*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB�*  �?
�
.Encoder/second_layer/batch_normalization/gamma
VariableV2*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
5Encoder/second_layer/batch_normalization/gamma/AssignAssign.Encoder/second_layer/batch_normalization/gamma?Encoder/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
3Encoder/second_layer/batch_normalization/gamma/readIdentity.Encoder/second_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
?Encoder/second_layer/batch_normalization/beta/Initializer/zerosConst*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
-Encoder/second_layer/batch_normalization/beta
VariableV2*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
4Encoder/second_layer/batch_normalization/beta/AssignAssign-Encoder/second_layer/batch_normalization/beta?Encoder/second_layer/batch_normalization/beta/Initializer/zeros*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
2Encoder/second_layer/batch_normalization/beta/readIdentity-Encoder/second_layer/batch_normalization/beta*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:�*
T0
�
FEncoder/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4Encoder/second_layer/batch_normalization/moving_mean
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
	container 
�
;Encoder/second_layer/batch_normalization/moving_mean/AssignAssign4Encoder/second_layer/batch_normalization/moving_meanFEncoder/second_layer/batch_normalization/moving_mean/Initializer/zeros*
T0*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
9Encoder/second_layer/batch_normalization/moving_mean/readIdentity4Encoder/second_layer/batch_normalization/moving_mean*
_output_shapes	
:�*
T0*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean
�
IEncoder/second_layer/batch_normalization/moving_variance/Initializer/onesConst*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
8Encoder/second_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
	container *
shape:�
�
?Encoder/second_layer/batch_normalization/moving_variance/AssignAssign8Encoder/second_layer/batch_normalization/moving_varianceIEncoder/second_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:�
�
=Encoder/second_layer/batch_normalization/moving_variance/readIdentity8Encoder/second_layer/batch_normalization/moving_variance*
_output_shapes	
:�*
T0*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance
}
8Encoder/second_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
6Encoder/second_layer/batch_normalization/batchnorm/addAdd=Encoder/second_layer/batch_normalization/moving_variance/read8Encoder/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:�
�
8Encoder/second_layer/batch_normalization/batchnorm/RsqrtRsqrt6Encoder/second_layer/batch_normalization/batchnorm/add*
_output_shapes	
:�*
T0
�
6Encoder/second_layer/batch_normalization/batchnorm/mulMul8Encoder/second_layer/batch_normalization/batchnorm/Rsqrt3Encoder/second_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
�
8Encoder/second_layer/batch_normalization/batchnorm/mul_1Mul,Encoder/second_layer/fully_connected/BiasAdd6Encoder/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
�
8Encoder/second_layer/batch_normalization/batchnorm/mul_2Mul9Encoder/second_layer/batch_normalization/moving_mean/read6Encoder/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:�*
T0
�
6Encoder/second_layer/batch_normalization/batchnorm/subSub2Encoder/second_layer/batch_normalization/beta/read8Encoder/second_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
8Encoder/second_layer/batch_normalization/batchnorm/add_1Add8Encoder/second_layer/batch_normalization/batchnorm/mul_16Encoder/second_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:����������*
T0
j
%Encoder/second_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
#Encoder/second_layer/leaky_relu/mulMul%Encoder/second_layer/leaky_relu/alpha8Encoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Encoder/second_layer/leaky_reluMaximum#Encoder/second_layer/leaky_relu/mul8Encoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
:Encoder/encoder_mu/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   
�
8Encoder/encoder_mu/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
�
8Encoder/encoder_mu/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *?��=*
dtype0*
_output_shapes
: 
�
BEncoder/encoder_mu/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Encoder/encoder_mu/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	�d*

seed *
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
seed2 
�
8Encoder/encoder_mu/kernel/Initializer/random_uniform/subSub8Encoder/encoder_mu/kernel/Initializer/random_uniform/max8Encoder/encoder_mu/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
: 
�
8Encoder/encoder_mu/kernel/Initializer/random_uniform/mulMulBEncoder/encoder_mu/kernel/Initializer/random_uniform/RandomUniform8Encoder/encoder_mu/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	�d
�
4Encoder/encoder_mu/kernel/Initializer/random_uniformAdd8Encoder/encoder_mu/kernel/Initializer/random_uniform/mul8Encoder/encoder_mu/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	�d
�
Encoder/encoder_mu/kernel
VariableV2*
dtype0*
_output_shapes
:	�d*
shared_name *,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container *
shape:	�d
�
 Encoder/encoder_mu/kernel/AssignAssignEncoder/encoder_mu/kernel4Encoder/encoder_mu/kernel/Initializer/random_uniform*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0
�
Encoder/encoder_mu/kernel/readIdentityEncoder/encoder_mu/kernel*
_output_shapes
:	�d*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel
�
)Encoder/encoder_mu/bias/Initializer/zerosConst*
_output_shapes
:d**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0
�
Encoder/encoder_mu/bias
VariableV2*
dtype0*
_output_shapes
:d*
shared_name **
_class 
loc:@Encoder/encoder_mu/bias*
	container *
shape:d
�
Encoder/encoder_mu/bias/AssignAssignEncoder/encoder_mu/bias)Encoder/encoder_mu/bias/Initializer/zeros*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
validate_shape(*
_output_shapes
:d*
use_locking(
�
Encoder/encoder_mu/bias/readIdentityEncoder/encoder_mu/bias**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d*
T0
�
Encoder/encoder_mu/MatMulMatMulEncoder/second_layer/leaky_reluEncoder/encoder_mu/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������d*
transpose_a( 
�
Encoder/encoder_mu/BiasAddBiasAddEncoder/encoder_mu/MatMulEncoder/encoder_mu/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������d
�
>Encoder/encoder_logvar/kernel/Initializer/random_uniform/shapeConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
�
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/minConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
�
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/maxConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *?��=*
dtype0*
_output_shapes
: 
�
FEncoder/encoder_logvar/kernel/Initializer/random_uniform/RandomUniformRandomUniform>Encoder/encoder_logvar/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	�d*

seed *
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
seed2 
�
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/subSub<Encoder/encoder_logvar/kernel/Initializer/random_uniform/max<Encoder/encoder_logvar/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
: 
�
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/mulMulFEncoder/encoder_logvar/kernel/Initializer/random_uniform/RandomUniform<Encoder/encoder_logvar/kernel/Initializer/random_uniform/sub*
_output_shapes
:	�d*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel
�
8Encoder/encoder_logvar/kernel/Initializer/random_uniformAdd<Encoder/encoder_logvar/kernel/Initializer/random_uniform/mul<Encoder/encoder_logvar/kernel/Initializer/random_uniform/min*
_output_shapes
:	�d*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel
�
Encoder/encoder_logvar/kernel
VariableV2*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d
�
$Encoder/encoder_logvar/kernel/AssignAssignEncoder/encoder_logvar/kernel8Encoder/encoder_logvar/kernel/Initializer/random_uniform*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0
�
"Encoder/encoder_logvar/kernel/readIdentityEncoder/encoder_logvar/kernel*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	�d
�
-Encoder/encoder_logvar/bias/Initializer/zerosConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0*
_output_shapes
:d
�
Encoder/encoder_logvar/bias
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container 
�
"Encoder/encoder_logvar/bias/AssignAssignEncoder/encoder_logvar/bias-Encoder/encoder_logvar/bias/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
:d
�
 Encoder/encoder_logvar/bias/readIdentityEncoder/encoder_logvar/bias*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
:d
�
Encoder/encoder_logvar/MatMulMatMulEncoder/second_layer/leaky_relu"Encoder/encoder_logvar/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
�
Encoder/encoder_logvar/BiasAddBiasAddEncoder/encoder_logvar/MatMul Encoder/encoder_logvar/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������d
e
Encoder/random_normal/shapeConst*
valueB	Rd*
dtype0	*
_output_shapes
:
_
Encoder/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
Encoder/random_normal/stddevConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
*Encoder/random_normal/RandomStandardNormalRandomStandardNormalEncoder/random_normal/shape*
_output_shapes
:d*
seed2 *

seed *
T0	*
dtype0
�
Encoder/random_normal/mulMul*Encoder/random_normal/RandomStandardNormalEncoder/random_normal/stddev*
_output_shapes
:d*
T0
x
Encoder/random_normalAddEncoder/random_normal/mulEncoder/random_normal/mean*
_output_shapes
:d*
T0
V
Encoder/truediv/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0

Encoder/truedivRealDivEncoder/encoder_logvar/BiasAddEncoder/truediv/y*'
_output_shapes
:���������d*
T0
U
Encoder/ExpExpEncoder/truediv*
T0*'
_output_shapes
:���������d
o
Encoder/logvar_stdMulEncoder/random_normalEncoder/Exp*
T0*'
_output_shapes
:���������d
t
Encoder/AddAddEncoder/logvar_stdEncoder/encoder_mu/BiasAdd*'
_output_shapes
:���������d*
T0
^
Encoder/encoder_codeSigmoidEncoder/Add*'
_output_shapes
:���������d*
T0
�
KDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
�
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *?�ʽ*
dtype0
�
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *?��=*
dtype0*
_output_shapes
: 
�
SDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformKDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d�*

seed *
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
seed2 
�
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
: 
�
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulSDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d�*
T0
�
EDecoder/first_layer/fully_connected/kernel/Initializer/random_uniformAddIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
*Decoder/first_layer/fully_connected/kernel
VariableV2*
shared_name *=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�
�
1Decoder/first_layer/fully_connected/kernel/AssignAssign*Decoder/first_layer/fully_connected/kernelEDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel
�
/Decoder/first_layer/fully_connected/kernel/readIdentity*Decoder/first_layer/fully_connected/kernel*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
:Decoder/first_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
valueB�*    
�
(Decoder/first_layer/fully_connected/bias
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias
�
/Decoder/first_layer/fully_connected/bias/AssignAssign(Decoder/first_layer/fully_connected/bias:Decoder/first_layer/fully_connected/bias/Initializer/zeros*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
-Decoder/first_layer/fully_connected/bias/readIdentity(Decoder/first_layer/fully_connected/bias*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
_output_shapes	
:�
�
*Decoder/first_layer/fully_connected/MatMulMatMulEncoder/encoder_code/Decoder/first_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
+Decoder/first_layer/fully_connected/BiasAddBiasAdd*Decoder/first_layer/fully_connected/MatMul-Decoder/first_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
i
$Decoder/first_layer/leaky_relu/alphaConst*
_output_shapes
: *
valueB
 *��L>*
dtype0
�
"Decoder/first_layer/leaky_relu/mulMul$Decoder/first_layer/leaky_relu/alpha+Decoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Decoder/first_layer/leaky_reluMaximum"Decoder/first_layer/leaky_relu/mul+Decoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
LDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *qĜ=*
dtype0
�
TDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
��*

seed *
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel
�
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel
�
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
FDecoder/second_layer/fully_connected/kernel/Initializer/random_uniformAddJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
+Decoder/second_layer/fully_connected/kernel
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
	container 
�
2Decoder/second_layer/fully_connected/kernel/AssignAssign+Decoder/second_layer/fully_connected/kernelFDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel
�
0Decoder/second_layer/fully_connected/kernel/readIdentity+Decoder/second_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
;Decoder/second_layer/fully_connected/bias/Initializer/zerosConst*
_output_shapes	
:�*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
valueB�*    *
dtype0
�
)Decoder/second_layer/fully_connected/bias
VariableV2*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
0Decoder/second_layer/fully_connected/bias/AssignAssign)Decoder/second_layer/fully_connected/bias;Decoder/second_layer/fully_connected/bias/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
validate_shape(
�
.Decoder/second_layer/fully_connected/bias/readIdentity)Decoder/second_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
_output_shapes	
:�
�
+Decoder/second_layer/fully_connected/MatMulMatMulDecoder/first_layer/leaky_relu0Decoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
,Decoder/second_layer/fully_connected/BiasAddBiasAdd+Decoder/second_layer/fully_connected/MatMul.Decoder/second_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
�
?Decoder/second_layer/batch_normalization/gamma/Initializer/onesConst*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
.Decoder/second_layer/batch_normalization/gamma
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
	container 
�
5Decoder/second_layer/batch_normalization/gamma/AssignAssign.Decoder/second_layer/batch_normalization/gamma?Decoder/second_layer/batch_normalization/gamma/Initializer/ones*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
validate_shape(
�
3Decoder/second_layer/batch_normalization/gamma/readIdentity.Decoder/second_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
?Decoder/second_layer/batch_normalization/beta/Initializer/zerosConst*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
-Decoder/second_layer/batch_normalization/beta
VariableV2*
shared_name *@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
4Decoder/second_layer/batch_normalization/beta/AssignAssign-Decoder/second_layer/batch_normalization/beta?Decoder/second_layer/batch_normalization/beta/Initializer/zeros*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
2Decoder/second_layer/batch_normalization/beta/readIdentity-Decoder/second_layer/batch_normalization/beta*
_output_shapes	
:�*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta
�
FDecoder/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4Decoder/second_layer/batch_normalization/moving_mean
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
	container 
�
;Decoder/second_layer/batch_normalization/moving_mean/AssignAssign4Decoder/second_layer/batch_normalization/moving_meanFDecoder/second_layer/batch_normalization/moving_mean/Initializer/zeros*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
9Decoder/second_layer/batch_normalization/moving_mean/readIdentity4Decoder/second_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
_output_shapes	
:�
�
IDecoder/second_layer/batch_normalization/moving_variance/Initializer/onesConst*
_output_shapes	
:�*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
valueB�*  �?*
dtype0
�
8Decoder/second_layer/batch_normalization/moving_variance
VariableV2*
shared_name *K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
?Decoder/second_layer/batch_normalization/moving_variance/AssignAssign8Decoder/second_layer/batch_normalization/moving_varianceIDecoder/second_layer/batch_normalization/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance
�
=Decoder/second_layer/batch_normalization/moving_variance/readIdentity8Decoder/second_layer/batch_normalization/moving_variance*
_output_shapes	
:�*
T0*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance
}
8Decoder/second_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
6Decoder/second_layer/batch_normalization/batchnorm/addAdd=Decoder/second_layer/batch_normalization/moving_variance/read8Decoder/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:�
�
8Decoder/second_layer/batch_normalization/batchnorm/RsqrtRsqrt6Decoder/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:�
�
6Decoder/second_layer/batch_normalization/batchnorm/mulMul8Decoder/second_layer/batch_normalization/batchnorm/Rsqrt3Decoder/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
8Decoder/second_layer/batch_normalization/batchnorm/mul_1Mul,Decoder/second_layer/fully_connected/BiasAdd6Decoder/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
8Decoder/second_layer/batch_normalization/batchnorm/mul_2Mul9Decoder/second_layer/batch_normalization/moving_mean/read6Decoder/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:�*
T0
�
6Decoder/second_layer/batch_normalization/batchnorm/subSub2Decoder/second_layer/batch_normalization/beta/read8Decoder/second_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
8Decoder/second_layer/batch_normalization/batchnorm/add_1Add8Decoder/second_layer/batch_normalization/batchnorm/mul_16Decoder/second_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:����������*
T0
j
%Decoder/second_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
#Decoder/second_layer/leaky_relu/mulMul%Decoder/second_layer/leaky_relu/alpha8Decoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Decoder/second_layer/leaky_reluMaximum#Decoder/second_layer/leaky_relu/mul8Decoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
5Decoder/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*'
_class
loc:@Decoder/dense/kernel*
valueB"     *
dtype0
�
3Decoder/dense/kernel/Initializer/random_uniform/minConst*'
_class
loc:@Decoder/dense/kernel*
valueB
 *HY��*
dtype0*
_output_shapes
: 
�
3Decoder/dense/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@Decoder/dense/kernel*
valueB
 *HY�=*
dtype0*
_output_shapes
: 
�
=Decoder/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5Decoder/dense/kernel/Initializer/random_uniform/shape*'
_class
loc:@Decoder/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
��*

seed *
T0
�
3Decoder/dense/kernel/Initializer/random_uniform/subSub3Decoder/dense/kernel/Initializer/random_uniform/max3Decoder/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@Decoder/dense/kernel
�
3Decoder/dense/kernel/Initializer/random_uniform/mulMul=Decoder/dense/kernel/Initializer/random_uniform/RandomUniform3Decoder/dense/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*'
_class
loc:@Decoder/dense/kernel
�
/Decoder/dense/kernel/Initializer/random_uniformAdd3Decoder/dense/kernel/Initializer/random_uniform/mul3Decoder/dense/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*'
_class
loc:@Decoder/dense/kernel
�
Decoder/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *'
_class
loc:@Decoder/dense/kernel*
	container *
shape:
��
�
Decoder/dense/kernel/AssignAssignDecoder/dense/kernel/Decoder/dense/kernel/Initializer/random_uniform* 
_output_shapes
:
��*
use_locking(*
T0*'
_class
loc:@Decoder/dense/kernel*
validate_shape(
�
Decoder/dense/kernel/readIdentityDecoder/dense/kernel* 
_output_shapes
:
��*
T0*'
_class
loc:@Decoder/dense/kernel
�
$Decoder/dense/bias/Initializer/zerosConst*%
_class
loc:@Decoder/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Decoder/dense/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *%
_class
loc:@Decoder/dense/bias*
	container *
shape:�
�
Decoder/dense/bias/AssignAssignDecoder/dense/bias$Decoder/dense/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes	
:�
�
Decoder/dense/bias/readIdentityDecoder/dense/bias*
_output_shapes	
:�*
T0*%
_class
loc:@Decoder/dense/bias
�
Decoder/dense/MatMulMatMulDecoder/second_layer/leaky_reluDecoder/dense/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
Decoder/dense/BiasAddBiasAddDecoder/dense/MatMulDecoder/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
d
Decoder/last_layerTanhDecoder/dense/BiasAdd*(
_output_shapes
:����������*
T0
t
Decoder/reshape_image/shapeConst*%
valueB"����         *
dtype0*
_output_shapes
:
�
Decoder/reshape_imageReshapeDecoder/last_layerDecoder/reshape_image/shape*
T0*
Tshape0*/
_output_shapes
:���������
~
Discriminator/noise_code_inPlaceholder*
dtype0*'
_output_shapes
:���������d*
shape:���������d
�
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *?��=*
dtype0
�
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d�*

seed *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
seed2 
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
: 
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
0Discriminator/first_layer/fully_connected/kernel
VariableV2*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�
�
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(
�
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d�*
T0
�
@Discriminator/first_layer/fully_connected/bias/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
.Discriminator/first_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:�
�
5Discriminator/first_layer/fully_connected/bias/AssignAssign.Discriminator/first_layer/fully_connected/bias@Discriminator/first_layer/fully_connected/bias/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(
�
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:�
�
0Discriminator/first_layer/fully_connected/MatMulMatMulEncoder/encoder_code5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
o
*Discriminator/first_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
$Discriminator/first_layer/leaky_reluMaximum(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
RDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *��=*
dtype0*
_output_shapes
: 
�
ZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformRDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
��*

seed *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
seed2 *
dtype0
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
_output_shapes
: 
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0
�
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
1Discriminator/second_layer/fully_connected/kernel
VariableV2* 
_output_shapes
:
��*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0
�
8Discriminator/second_layer/fully_connected/kernel/AssignAssign1Discriminator/second_layer/fully_connected/kernelLDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
��*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(
�
6Discriminator/second_layer/fully_connected/kernel/readIdentity1Discriminator/second_layer/fully_connected/kernel*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
ADiscriminator/second_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    
�
/Discriminator/second_layer/fully_connected/bias
VariableV2*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
6Discriminator/second_layer/fully_connected/bias/AssignAssign/Discriminator/second_layer/fully_connected/biasADiscriminator/second_layer/fully_connected/bias/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(
�
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
2Discriminator/second_layer/fully_connected/BiasAddBiasAdd1Discriminator/second_layer/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
p
+Discriminator/second_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
)Discriminator/second_layer/leaky_relu/mulMul+Discriminator/second_layer/leaky_relu/alpha2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
%Discriminator/second_layer/leaky_reluMaximum)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
:Discriminator/prob/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@Discriminator/prob/kernel*
valueB"      
�
8Discriminator/prob/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Iv�
�
8Discriminator/prob/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
�
BDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Discriminator/prob/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	�*

seed *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
seed2 
�
8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*,
_class"
 loc:@Discriminator/prob/kernel
�
8Discriminator/prob/kernel/Initializer/random_uniform/mulMulBDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniform8Discriminator/prob/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
4Discriminator/prob/kernel/Initializer/random_uniformAdd8Discriminator/prob/kernel/Initializer/random_uniform/mul8Discriminator/prob/kernel/Initializer/random_uniform/min*
_output_shapes
:	�*
T0*,
_class"
 loc:@Discriminator/prob/kernel
�
Discriminator/prob/kernel
VariableV2*
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container 
�
 Discriminator/prob/kernel/AssignAssignDiscriminator/prob/kernel4Discriminator/prob/kernel/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�
�
Discriminator/prob/kernel/readIdentityDiscriminator/prob/kernel*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
)Discriminator/prob/bias/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
�
Discriminator/prob/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias
�
Discriminator/prob/bias/AssignAssignDiscriminator/prob/bias)Discriminator/prob/bias/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
�
Discriminator/prob/bias/readIdentityDiscriminator/prob/bias**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:*
T0
�
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
�
2Discriminator/first_layer_1/fully_connected/MatMulMatMulDiscriminator/noise_code_in5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
3Discriminator/first_layer_1/fully_connected/BiasAddBiasAdd2Discriminator/first_layer_1/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
q
,Discriminator/first_layer_1/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
*Discriminator/first_layer_1/leaky_relu/mulMul,Discriminator/first_layer_1/leaky_relu/alpha3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
&Discriminator/first_layer_1/leaky_reluMaximum*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
3Discriminator/second_layer_1/fully_connected/MatMulMatMul&Discriminator/first_layer_1/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
4Discriminator/second_layer_1/fully_connected/BiasAddBiasAdd3Discriminator/second_layer_1/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
r
-Discriminator/second_layer_1/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
+Discriminator/second_layer_1/leaky_relu/mulMul-Discriminator/second_layer_1/leaky_relu/alpha4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
'Discriminator/second_layer_1/leaky_reluMaximum+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Discriminator/prob_1/MatMulMatMul'Discriminator/second_layer_1/leaky_reluDiscriminator/prob/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
Discriminator/prob_1/BiasAddBiasAddDiscriminator/prob_1/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
i
ones_like/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
T
ones_like/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
w
	ones_likeFillones_like/Shapeones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
s
logistic_loss/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:���������*
T0
�
logistic_loss/SelectSelectlogistic_loss/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:���������*
T0
f
logistic_loss/NegNegDiscriminator/prob/BiasAdd*'
_output_shapes
:���������*
T0
�
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/NegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
q
logistic_loss/mulMulDiscriminator/prob/BiasAdd	ones_like*'
_output_shapes
:���������*
T0
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0*'
_output_shapes
:���������
b
logistic_loss/ExpExplogistic_loss/Select_1*
T0*'
_output_shapes
:���������
a
logistic_loss/Log1pLog1plogistic_loss/Exp*'
_output_shapes
:���������*
T0
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*'
_output_shapes
:���������*
T0
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
m
adversalrial_lossMeanlogistic_lossConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
l
SubSubDecoder/reshape_imageEncoder/real_in*/
_output_shapes
:���������*
T0
I
AbsAbsSub*
T0*/
_output_shapes
:���������
`
Const_1Const*%
valueB"             *
dtype0*
_output_shapes
:
b
pixelwise_lossMeanAbsConst_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
J
mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
E
mulMulmul/xadversalrial_loss*
T0*
_output_shapes
: 
L
mul_1/xConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
F
mul_1Mulmul_1/xpixelwise_loss*
T0*
_output_shapes
: 
B
generator_lossAddmulmul_1*
T0*
_output_shapes
: 
e
generator_loss_1/tagConst*!
valueB Bgenerator_loss_1*
dtype0*
_output_shapes
: 
k
generator_loss_1HistogramSummarygenerator_loss_1/taggenerator_loss*
T0*
_output_shapes
: 
m
ones_like_1/ShapeShapeDiscriminator/prob_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
V
ones_like_1/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
}
ones_like_1Fillones_like_1/Shapeones_like_1/Const*
T0*

index_type0*'
_output_shapes
:���������
w
logistic_loss_1/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss_1/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:���������
�
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:���������
j
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:���������*
T0
�
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
w
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAddones_like_1*
T0*'
_output_shapes
:���������
y
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*
T0*'
_output_shapes
:���������
f
logistic_loss_1/ExpExplogistic_loss_1/Select_1*
T0*'
_output_shapes
:���������
e
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*'
_output_shapes
:���������*
T0
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*
T0*'
_output_shapes
:���������
X
Const_2Const*
_output_shapes
:*
valueB"       *
dtype0
d
MeanMeanlogistic_loss_1Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e

zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
u
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:���������*
T0
�
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:���������*
T0
h
logistic_loss_2/NegNegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
t
logistic_loss_2/mulMulDiscriminator/prob/BiasAdd
zeros_like*'
_output_shapes
:���������*
T0
y
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*
T0*'
_output_shapes
:���������
f
logistic_loss_2/ExpExplogistic_loss_2/Select_1*
T0*'
_output_shapes
:���������
e
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*
T0*'
_output_shapes
:���������
t
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*
T0*'
_output_shapes
:���������
X
Const_3Const*
dtype0*
_output_shapes
:*
valueB"       
f
Mean_1Meanlogistic_loss_2Const_3*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
9
addAddMeanMean_1*
T0*
_output_shapes
: 
i
discriminator_loss/tagConst*#
valueB Bdiscriminator_loss*
dtype0*
_output_shapes
: 
d
discriminator_lossHistogramSummarydiscriminator_loss/tagadd*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
<
#gradients/add_grad/tuple/group_depsNoOp^gradients/Fill
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Fill$^gradients/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/Fill$^gradients/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshape+gradients/add_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
h
gradients/Mean_grad/ShapeShapelogistic_loss_1*
out_type0*
_output_shapes
:*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
j
gradients/Mean_grad/Shape_1Shapelogistic_loss_1*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:���������
t
#gradients/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
gradients/Mean_1_grad/ReshapeReshape-gradients/add_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
j
gradients/Mean_1_grad/ShapeShapelogistic_loss_2*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
l
gradients/Mean_1_grad/Shape_1Shapelogistic_loss_2*
T0*
out_type0*
_output_shapes
:
`
gradients/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
e
gradients/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
g
gradients/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
a
gradients/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
_output_shapes
: *
T0
�
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*'
_output_shapes
:���������*
T0
w
$gradients/logistic_loss_1_grad/ShapeShapelogistic_loss_1/sub*
out_type0*
_output_shapes
:*
T0
{
&gradients/logistic_loss_1_grad/Shape_1Shapelogistic_loss_1/Log1p*
T0*
out_type0*
_output_shapes
:
�
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
/gradients/logistic_loss_1_grad/tuple/group_depsNoOp'^gradients/logistic_loss_1_grad/Reshape)^gradients/logistic_loss_1_grad/Reshape_1
�
7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape
�
9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1
w
$gradients/logistic_loss_2_grad/ShapeShapelogistic_loss_2/sub*
_output_shapes
:*
T0*
out_type0
{
&gradients/logistic_loss_2_grad/Shape_1Shapelogistic_loss_2/Log1p*
T0*
out_type0*
_output_shapes
:
�
4gradients/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_2_grad/Shape&gradients/logistic_loss_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"gradients/logistic_loss_2_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
&gradients/logistic_loss_2_grad/ReshapeReshape"gradients/logistic_loss_2_grad/Sum$gradients/logistic_loss_2_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
$gradients/logistic_loss_2_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(gradients/logistic_loss_2_grad/Reshape_1Reshape$gradients/logistic_loss_2_grad/Sum_1&gradients/logistic_loss_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/gradients/logistic_loss_2_grad/tuple/group_depsNoOp'^gradients/logistic_loss_2_grad/Reshape)^gradients/logistic_loss_2_grad/Reshape_1
�
7gradients/logistic_loss_2_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_2_grad/Reshape0^gradients/logistic_loss_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*9
_class/
-+loc:@gradients/logistic_loss_2_grad/Reshape
�
9gradients/logistic_loss_2_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_2_grad/Reshape_10^gradients/logistic_loss_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@gradients/logistic_loss_2_grad/Reshape_1
~
(gradients/logistic_loss_1/sub_grad/ShapeShapelogistic_loss_1/Select*
out_type0*
_output_shapes
:*
T0
}
*gradients/logistic_loss_1/sub_grad/Shape_1Shapelogistic_loss_1/mul*
_output_shapes
:*
T0*
out_type0
�
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients/logistic_loss_1/sub_grad/Sum_1Sum7gradients/logistic_loss_1_grad/tuple/control_dependency:gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
z
&gradients/logistic_loss_1/sub_grad/NegNeg(gradients/logistic_loss_1/sub_grad/Sum_1*
T0*
_output_shapes
:
�
,gradients/logistic_loss_1/sub_grad/Reshape_1Reshape&gradients/logistic_loss_1/sub_grad/Neg*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
3gradients/logistic_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/sub_grad/Reshape-^gradients/logistic_loss_1/sub_grad/Reshape_1
�
;gradients/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/sub_grad/Reshape4^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/sub_grad/Reshape*'
_output_shapes
:���������
�
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1*'
_output_shapes
:���������
�
*gradients/logistic_loss_1/Log1p_grad/add/xConst:^gradients/logistic_loss_1_grad/tuple/control_dependency_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
(gradients/logistic_loss_1/Log1p_grad/addAdd*gradients/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*
T0*'
_output_shapes
:���������
�
/gradients/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*
T0*'
_output_shapes
:���������
�
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:���������
~
(gradients/logistic_loss_2/sub_grad/ShapeShapelogistic_loss_2/Select*
T0*
out_type0*
_output_shapes
:
}
*gradients/logistic_loss_2/sub_grad/Shape_1Shapelogistic_loss_2/mul*
T0*
out_type0*
_output_shapes
:
�
8gradients/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_2/sub_grad/Shape*gradients/logistic_loss_2/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients/logistic_loss_2/sub_grad/SumSum7gradients/logistic_loss_2_grad/tuple/control_dependency8gradients/logistic_loss_2/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients/logistic_loss_2/sub_grad/ReshapeReshape&gradients/logistic_loss_2/sub_grad/Sum(gradients/logistic_loss_2/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients/logistic_loss_2/sub_grad/Sum_1Sum7gradients/logistic_loss_2_grad/tuple/control_dependency:gradients/logistic_loss_2/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
z
&gradients/logistic_loss_2/sub_grad/NegNeg(gradients/logistic_loss_2/sub_grad/Sum_1*
T0*
_output_shapes
:
�
,gradients/logistic_loss_2/sub_grad/Reshape_1Reshape&gradients/logistic_loss_2/sub_grad/Neg*gradients/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
3gradients/logistic_loss_2/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_2/sub_grad/Reshape-^gradients/logistic_loss_2/sub_grad/Reshape_1
�
;gradients/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_2/sub_grad/Reshape4^gradients/logistic_loss_2/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_2/sub_grad/Reshape*'
_output_shapes
:���������
�
=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_2/sub_grad/Reshape_14^gradients/logistic_loss_2/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/sub_grad/Reshape_1*'
_output_shapes
:���������
�
*gradients/logistic_loss_2/Log1p_grad/add/xConst:^gradients/logistic_loss_2_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
(gradients/logistic_loss_2/Log1p_grad/addAdd*gradients/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*
T0*'
_output_shapes
:���������
�
/gradients/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_2/Log1p_grad/add*
T0*'
_output_shapes
:���������
�
(gradients/logistic_loss_2/Log1p_grad/mulMul9gradients/logistic_loss_2_grad/tuple/control_dependency_1/gradients/logistic_loss_2/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
0gradients/logistic_loss_1/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
�
,gradients/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual;gradients/logistic_loss_1/sub_grad/tuple/control_dependency0gradients/logistic_loss_1/Select_grad/zeros_like*'
_output_shapes
:���������*
T0
�
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
�
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:���������
�
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*A
_class7
53loc:@gradients/logistic_loss_1/Select_grad/Select_1*'
_output_shapes
:���������*
T0
�
(gradients/logistic_loss_1/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
out_type0*
_output_shapes
:*
T0
u
*gradients/logistic_loss_1/mul_grad/Shape_1Shapeones_like_1*
T0*
out_type0*
_output_shapes
:
�
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1ones_like_1*
T0*'
_output_shapes
:���������
�
&gradients/logistic_loss_1/mul_grad/SumSum&gradients/logistic_loss_1/mul_grad/Mul8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients/logistic_loss_1/mul_grad/ReshapeReshape&gradients/logistic_loss_1/mul_grad/Sum(gradients/logistic_loss_1/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients/logistic_loss_1/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
3gradients/logistic_loss_1/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/mul_grad/Reshape-^gradients/logistic_loss_1/mul_grad/Reshape_1
�
;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/mul_grad/Reshape4^gradients/logistic_loss_1/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/mul_grad/Reshape
�
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1*'
_output_shapes
:���������
�
&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0*'
_output_shapes
:���������
�
0gradients/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:���������*
T0
�
,gradients/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual;gradients/logistic_loss_2/sub_grad/tuple/control_dependency0gradients/logistic_loss_2/Select_grad/zeros_like*
T0*'
_output_shapes
:���������
�
.gradients/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual0gradients/logistic_loss_2/Select_grad/zeros_like;gradients/logistic_loss_2/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
6gradients/logistic_loss_2/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_2/Select_grad/Select/^gradients/logistic_loss_2/Select_grad/Select_1
�
>gradients/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_2/Select_grad/Select7^gradients/logistic_loss_2/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*'
_output_shapes
:���������
�
@gradients/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_2/Select_grad/Select_17^gradients/logistic_loss_2/Select_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@gradients/logistic_loss_2/Select_grad/Select_1
�
(gradients/logistic_loss_2/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
t
*gradients/logistic_loss_2/mul_grad/Shape_1Shape
zeros_like*
T0*
out_type0*
_output_shapes
:
�
8gradients/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_2/mul_grad/Shape*gradients/logistic_loss_2/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
&gradients/logistic_loss_2/mul_grad/MulMul=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1
zeros_like*
T0*'
_output_shapes
:���������
�
&gradients/logistic_loss_2/mul_grad/SumSum&gradients/logistic_loss_2/mul_grad/Mul8gradients/logistic_loss_2/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*gradients/logistic_loss_2/mul_grad/ReshapeReshape&gradients/logistic_loss_2/mul_grad/Sum(gradients/logistic_loss_2/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
(gradients/logistic_loss_2/mul_grad/Sum_1Sum(gradients/logistic_loss_2/mul_grad/Mul_1:gradients/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
,gradients/logistic_loss_2/mul_grad/Reshape_1Reshape(gradients/logistic_loss_2/mul_grad/Sum_1*gradients/logistic_loss_2/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
3gradients/logistic_loss_2/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_2/mul_grad/Reshape-^gradients/logistic_loss_2/mul_grad/Reshape_1
�
;gradients/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_2/mul_grad/Reshape4^gradients/logistic_loss_2/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_2/mul_grad/Reshape*'
_output_shapes
:���������
�
=gradients/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_2/mul_grad/Reshape_14^gradients/logistic_loss_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:���������
�
&gradients/logistic_loss_2/Exp_grad/mulMul(gradients/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*
T0*'
_output_shapes
:���������
�
2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*'
_output_shapes
:���������*
T0
�
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*'
_output_shapes
:���������*
T0
�
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*'
_output_shapes
:���������*
T0
�
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
�
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select
�
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1*'
_output_shapes
:���������
�
2gradients/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*
T0*'
_output_shapes
:���������
�
.gradients/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual&gradients/logistic_loss_2/Exp_grad/mul2gradients/logistic_loss_2/Select_1_grad/zeros_like*
T0*'
_output_shapes
:���������
�
0gradients/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients/logistic_loss_2/Select_1_grad/zeros_like&gradients/logistic_loss_2/Exp_grad/mul*'
_output_shapes
:���������*
T0
�
8gradients/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_2/Select_1_grad/Select1^gradients/logistic_loss_2/Select_1_grad/Select_1
�
@gradients/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_2/Select_1_grad/Select9^gradients/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_2/Select_1_grad/Select*'
_output_shapes
:���������
�
Bgradients/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_2/Select_1_grad/Select_19^gradients/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_2/Select_1_grad/Select_1*'
_output_shapes
:���������
�
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
&gradients/logistic_loss_2/Neg_grad/NegNeg@gradients/logistic_loss_2/Select_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
gradients/AddNAddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N*'
_output_shapes
:���������
�
7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:
�
<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN8^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:���������
�
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
gradients/AddN_1AddN>gradients/logistic_loss_2/Select_grad/tuple/control_dependency;gradients/logistic_loss_2/mul_grad/tuple/control_dependencyBgradients/logistic_loss_2/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_2/Neg_grad/Neg*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*
N*'
_output_shapes
:���������*
T0
�
5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
_output_shapes
:*
T0*
data_formatNHWC
�
:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_16^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad
�
Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*'
_output_shapes
:���������
�
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
�
;gradients/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp2^gradients/Discriminator/prob_1/MatMul_grad/MatMul4^gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
�
Cgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/Discriminator/prob_1/MatMul_grad/MatMul<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
1gradients/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
9gradients/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp0^gradients/Discriminator/prob/MatMul_grad/MatMul2^gradients/Discriminator/prob/MatMul_grad/MatMul_1
�
Agradients/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity/gradients/Discriminator/prob/MatMul_grad/MatMul:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/Discriminator/prob/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	�*
T0
�
gradients/AddN_2AddNFgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:*
T0
�
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
�
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosFill>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Cgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape:gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape
�
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
:gradients/Discriminator/second_layer/leaky_relu_grad/zerosFill<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Jgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Discriminator/second_layer/leaky_relu_grad/Shape<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
;gradients/Discriminator/second_layer/leaky_relu_grad/SelectSelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency:gradients/Discriminator/second_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
8gradients/Discriminator/second_layer/leaky_relu_grad/SumSum;gradients/Discriminator/second_layer/leaky_relu_grad/SelectJgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1Lgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Egradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp=^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape?^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
�
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
�
gradients/AddN_3AddNEgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1*
N*
_output_shapes
:	�*
T0
�
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
Pgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Rgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Dgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1Reshape@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Kgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeE^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
�
Sgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeL^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*U
_classK
IGloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
�
Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1L^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Ngradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulMulMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Pgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Bgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Igradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpA^gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeC^gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
�
Qgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeJ^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_4AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
gradients/AddN_5AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
�
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
_output_shapes	
:�*
T0*
data_formatNHWC
�
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
�
\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Kgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Sgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulL^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
�
[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulT^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul
�
]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
�
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Igradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_reluZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
�
Qgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpH^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulJ^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
�
Ygradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityGgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulR^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
�
gradients/AddN_6AddN^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
;gradients/Discriminator/first_layer_1/leaky_relu_grad/zerosFill=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
9gradients/Discriminator/first_layer_1/leaky_relu_grad/SumSum<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectKgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Fgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp>^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape@^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
Ngradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeG^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*P
_classF
DBloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:����������*
T0
�
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
9gradients/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2ShapeYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
9gradients/Discriminator/first_layer/leaky_relu_grad/zerosFill;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
:gradients/Discriminator/first_layer/leaky_relu_grad/SelectSelect@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency9gradients/Discriminator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Select@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqual9gradients/Discriminator/first_layer/leaky_relu_grad/zerosYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
7gradients/Discriminator/first_layer/leaky_relu_grad/SumSum:gradients/Discriminator/first_layer/leaky_relu_grad/SelectIgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape7gradients/Discriminator/first_layer/leaky_relu_grad/Sum9gradients/Discriminator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Kgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Dgradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp<^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape>^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
�
Lgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeE^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*N
_classD
B@loc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape
�
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
gradients/AddN_7AddN]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
��
�
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Ogradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulOgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Qgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Cgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1Reshape?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Jgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeD^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
�
Rgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeK^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
�
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Ogradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Agradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Hgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp@^gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeB^gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
�
Pgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeI^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape
�
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
gradients/AddN_8AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
gradients/AddN_9AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������*
T0
�
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradR^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(
�
Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/noise_code_in[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d�*
transpose_a(*
transpose_b( *
T0
�
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
�
Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1S^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d�
�
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
�
Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_codeYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d�*
transpose_a(*
transpose_b( 
�
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d*
T0
�
Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d�
�
gradients/AddN_10AddN]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
gradients/AddN_11AddN\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1*
N*
_output_shapes
:	d�
�
beta1_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
�
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
beta2_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
�
beta2_power/readIdentitybeta2_power*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
�
eDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
�
[Discriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/ConstConst*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
�
UDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zerosFilleDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensor[Discriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
CDiscriminator/first_layer/fully_connected/kernel/discriminator_opti
VariableV2*
_output_shapes
:	d�*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:	d�*
dtype0
�
JDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/AssignAssignCDiscriminator/first_layer/fully_connected/kernel/discriminator_optiUDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(
�
HDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/readIdentityCDiscriminator/first_layer/fully_connected/kernel/discriminator_opti*
_output_shapes
:	d�*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
gDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
�
]Discriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/ConstConst*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
�
WDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zerosFillgDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensor]Discriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
EDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1
VariableV2*
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container 
�
LDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/AssignAssignEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1WDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0
�
JDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/readIdentityEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
SDiscriminator/first_layer/fully_connected/bias/discriminator_opti/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
ADiscriminator/first_layer/fully_connected/bias/discriminator_opti
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
HDiscriminator/first_layer/fully_connected/bias/discriminator_opti/AssignAssignADiscriminator/first_layer/fully_connected/bias/discriminator_optiSDiscriminator/first_layer/fully_connected/bias/discriminator_opti/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
FDiscriminator/first_layer/fully_connected/bias/discriminator_opti/readIdentityADiscriminator/first_layer/fully_connected/bias/discriminator_opti*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:�
�
UDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    
�
CDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1
VariableV2*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
JDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/AssignAssignCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1UDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
HDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/readIdentityCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1*
_output_shapes	
:�*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
fDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
\Discriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/ConstConst*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0
�
VDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zerosFillfDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensor\Discriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0
�
DDiscriminator/second_layer/fully_connected/kernel/discriminator_opti
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
KDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/AssignAssignDDiscriminator/second_layer/fully_connected/kernel/discriminator_optiVDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
IDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/readIdentityDDiscriminator/second_layer/fully_connected/kernel/discriminator_opti*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
hDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      
�
^Discriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
XDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zerosFillhDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensor^Discriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
FDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
MDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/AssignAssignFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1XDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
KDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/readIdentityFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
TDiscriminator/second_layer/fully_connected/bias/discriminator_opti/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
BDiscriminator/second_layer/fully_connected/bias/discriminator_opti
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
IDiscriminator/second_layer/fully_connected/bias/discriminator_opti/AssignAssignBDiscriminator/second_layer/fully_connected/bias/discriminator_optiTDiscriminator/second_layer/fully_connected/bias/discriminator_opti/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
GDiscriminator/second_layer/fully_connected/bias/discriminator_opti/readIdentityBDiscriminator/second_layer/fully_connected/bias/discriminator_opti*
_output_shapes	
:�*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
VDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
DDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1
VariableV2*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
KDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/AssignAssignDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1VDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
IDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/readIdentityDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:�*
T0
�
>Discriminator/prob/kernel/discriminator_opti/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
,Discriminator/prob/kernel/discriminator_opti
VariableV2*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
�
3Discriminator/prob/kernel/discriminator_opti/AssignAssign,Discriminator/prob/kernel/discriminator_opti>Discriminator/prob/kernel/discriminator_opti/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�
�
1Discriminator/prob/kernel/discriminator_opti/readIdentity,Discriminator/prob/kernel/discriminator_opti*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
@Discriminator/prob/kernel/discriminator_opti_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
.Discriminator/prob/kernel/discriminator_opti_1
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	�
�
5Discriminator/prob/kernel/discriminator_opti_1/AssignAssign.Discriminator/prob/kernel/discriminator_opti_1@Discriminator/prob/kernel/discriminator_opti_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�
�
3Discriminator/prob/kernel/discriminator_opti_1/readIdentity.Discriminator/prob/kernel/discriminator_opti_1*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
<Discriminator/prob/bias/discriminator_opti/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
�
*Discriminator/prob/bias/discriminator_opti
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container 
�
1Discriminator/prob/bias/discriminator_opti/AssignAssign*Discriminator/prob/bias/discriminator_opti<Discriminator/prob/bias/discriminator_opti/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
�
/Discriminator/prob/bias/discriminator_opti/readIdentity*Discriminator/prob/bias/discriminator_opti*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
�
>Discriminator/prob/bias/discriminator_opti_1/Initializer/zerosConst*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0
�
,Discriminator/prob/bias/discriminator_opti_1
VariableV2*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
3Discriminator/prob/bias/discriminator_opti_1/AssignAssign,Discriminator/prob/bias/discriminator_opti_1>Discriminator/prob/bias/discriminator_opti_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(
�
1Discriminator/prob/bias/discriminator_opti_1/readIdentity,Discriminator/prob/bias/discriminator_opti_1*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
e
 discriminator_opti/learning_rateConst*
valueB
 *�Q9*
dtype0*
_output_shapes
: 
]
discriminator_opti/beta1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
]
discriminator_opti/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
_
discriminator_opti/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Tdiscriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam0Discriminator/first_layer/fully_connected/kernelCDiscriminator/first_layer/fully_connected/kernel/discriminator_optiEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_11*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	d�*
use_locking( 
�
Rdiscriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam.Discriminator/first_layer/fully_connected/biasADiscriminator/first_layer/fully_connected/bias/discriminator_optiCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_10*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
Udiscriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernelDDiscriminator/second_layer/fully_connected/kernel/discriminator_optiFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_7*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
Sdiscriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/biasBDiscriminator/second_layer/fully_connected/bias/discriminator_optiDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_6*
use_locking( *
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
=discriminator_opti/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernel,Discriminator/prob/kernel/discriminator_opti.Discriminator/prob/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_3*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
use_nesterov( *
_output_shapes
:	�
�
;discriminator_opti/update_Discriminator/prob/bias/ApplyAdam	ApplyAdamDiscriminator/prob/bias*Discriminator/prob/bias/discriminator_opti,Discriminator/prob/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_2*
use_locking( *
T0**
_class 
loc:@Discriminator/prob/bias*
use_nesterov( *
_output_shapes
:
�
discriminator_opti/mulMulbeta1_power/readdiscriminator_opti/beta1S^discriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamU^discriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam<^discriminator_opti/update_Discriminator/prob/bias/ApplyAdam>^discriminator_opti/update_Discriminator/prob/kernel/ApplyAdamT^discriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamV^discriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
�
discriminator_opti/AssignAssignbeta1_powerdiscriminator_opti/mul*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
�
discriminator_opti/mul_1Mulbeta2_power/readdiscriminator_opti/beta2S^discriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamU^discriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam<^discriminator_opti/update_Discriminator/prob/bias/ApplyAdam>^discriminator_opti/update_Discriminator/prob/kernel/ApplyAdamT^discriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamV^discriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
�
discriminator_opti/Assign_1Assignbeta2_powerdiscriminator_opti/mul_1*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
�
discriminator_optiNoOp^discriminator_opti/Assign^discriminator_opti/Assign_1S^discriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamU^discriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam<^discriminator_opti/update_Discriminator/prob/bias/ApplyAdam>^discriminator_opti/update_Discriminator/prob/kernel/ApplyAdamT^discriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamV^discriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*

index_type0*
_output_shapes
: *
T0
K
0gradients_1/generator_loss_grad/tuple/group_depsNoOp^gradients_1/Fill
�
8gradients_1/generator_loss_grad/tuple/control_dependencyIdentitygradients_1/Fill1^gradients_1/generator_loss_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill*
_output_shapes
: 
�
:gradients_1/generator_loss_grad/tuple/control_dependency_1Identitygradients_1/Fill1^gradients_1/generator_loss_grad/tuple/group_deps*#
_class
loc:@gradients_1/Fill*
_output_shapes
: *
T0
�
gradients_1/mul_grad/MulMul8gradients_1/generator_loss_grad/tuple/control_dependencyadversalrial_loss*
T0*
_output_shapes
: 
�
gradients_1/mul_grad/Mul_1Mul8gradients_1/generator_loss_grad/tuple/control_dependencymul/x*
T0*
_output_shapes
: 
e
%gradients_1/mul_grad/tuple/group_depsNoOp^gradients_1/mul_grad/Mul^gradients_1/mul_grad/Mul_1
�
-gradients_1/mul_grad/tuple/control_dependencyIdentitygradients_1/mul_grad/Mul&^gradients_1/mul_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients_1/mul_grad/Mul*
_output_shapes
: 
�
/gradients_1/mul_grad/tuple/control_dependency_1Identitygradients_1/mul_grad/Mul_1&^gradients_1/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_grad/Mul_1*
_output_shapes
: 
�
gradients_1/mul_1_grad/MulMul:gradients_1/generator_loss_grad/tuple/control_dependency_1pixelwise_loss*
_output_shapes
: *
T0
�
gradients_1/mul_1_grad/Mul_1Mul:gradients_1/generator_loss_grad/tuple/control_dependency_1mul_1/x*
_output_shapes
: *
T0
k
'gradients_1/mul_1_grad/tuple/group_depsNoOp^gradients_1/mul_1_grad/Mul^gradients_1/mul_1_grad/Mul_1
�
/gradients_1/mul_1_grad/tuple/control_dependencyIdentitygradients_1/mul_1_grad/Mul(^gradients_1/mul_1_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_1_grad/Mul*
_output_shapes
: 
�
1gradients_1/mul_1_grad/tuple/control_dependency_1Identitygradients_1/mul_1_grad/Mul_1(^gradients_1/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_1_grad/Mul_1*
_output_shapes
: 
�
0gradients_1/adversalrial_loss_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
*gradients_1/adversalrial_loss_grad/ReshapeReshape/gradients_1/mul_grad/tuple/control_dependency_10gradients_1/adversalrial_loss_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
u
(gradients_1/adversalrial_loss_grad/ShapeShapelogistic_loss*
T0*
out_type0*
_output_shapes
:
�
'gradients_1/adversalrial_loss_grad/TileTile*gradients_1/adversalrial_loss_grad/Reshape(gradients_1/adversalrial_loss_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
w
*gradients_1/adversalrial_loss_grad/Shape_1Shapelogistic_loss*
_output_shapes
:*
T0*
out_type0
m
*gradients_1/adversalrial_loss_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
r
(gradients_1/adversalrial_loss_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
'gradients_1/adversalrial_loss_grad/ProdProd*gradients_1/adversalrial_loss_grad/Shape_1(gradients_1/adversalrial_loss_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
t
*gradients_1/adversalrial_loss_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
)gradients_1/adversalrial_loss_grad/Prod_1Prod*gradients_1/adversalrial_loss_grad/Shape_2*gradients_1/adversalrial_loss_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
n
,gradients_1/adversalrial_loss_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
*gradients_1/adversalrial_loss_grad/MaximumMaximum)gradients_1/adversalrial_loss_grad/Prod_1,gradients_1/adversalrial_loss_grad/Maximum/y*
_output_shapes
: *
T0
�
+gradients_1/adversalrial_loss_grad/floordivFloorDiv'gradients_1/adversalrial_loss_grad/Prod*gradients_1/adversalrial_loss_grad/Maximum*
_output_shapes
: *
T0
�
'gradients_1/adversalrial_loss_grad/CastCast+gradients_1/adversalrial_loss_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
*gradients_1/adversalrial_loss_grad/truedivRealDiv'gradients_1/adversalrial_loss_grad/Tile'gradients_1/adversalrial_loss_grad/Cast*
T0*'
_output_shapes
:���������
�
-gradients_1/pixelwise_loss_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
�
'gradients_1/pixelwise_loss_grad/ReshapeReshape1gradients_1/mul_1_grad/tuple/control_dependency_1-gradients_1/pixelwise_loss_grad/Reshape/shape*&
_output_shapes
:*
T0*
Tshape0
h
%gradients_1/pixelwise_loss_grad/ShapeShapeAbs*
T0*
out_type0*
_output_shapes
:
�
$gradients_1/pixelwise_loss_grad/TileTile'gradients_1/pixelwise_loss_grad/Reshape%gradients_1/pixelwise_loss_grad/Shape*/
_output_shapes
:���������*

Tmultiples0*
T0
j
'gradients_1/pixelwise_loss_grad/Shape_1ShapeAbs*
T0*
out_type0*
_output_shapes
:
j
'gradients_1/pixelwise_loss_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%gradients_1/pixelwise_loss_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$gradients_1/pixelwise_loss_grad/ProdProd'gradients_1/pixelwise_loss_grad/Shape_1%gradients_1/pixelwise_loss_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
q
'gradients_1/pixelwise_loss_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&gradients_1/pixelwise_loss_grad/Prod_1Prod'gradients_1/pixelwise_loss_grad/Shape_2'gradients_1/pixelwise_loss_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)gradients_1/pixelwise_loss_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'gradients_1/pixelwise_loss_grad/MaximumMaximum&gradients_1/pixelwise_loss_grad/Prod_1)gradients_1/pixelwise_loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
(gradients_1/pixelwise_loss_grad/floordivFloorDiv$gradients_1/pixelwise_loss_grad/Prod'gradients_1/pixelwise_loss_grad/Maximum*
T0*
_output_shapes
: 
�
$gradients_1/pixelwise_loss_grad/CastCast(gradients_1/pixelwise_loss_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0*
Truncate( 
�
'gradients_1/pixelwise_loss_grad/truedivRealDiv$gradients_1/pixelwise_loss_grad/Tile$gradients_1/pixelwise_loss_grad/Cast*/
_output_shapes
:���������*
T0
u
$gradients_1/logistic_loss_grad/ShapeShapelogistic_loss/sub*
_output_shapes
:*
T0*
out_type0
y
&gradients_1/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:
�
4gradients_1/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_1/logistic_loss_grad/Shape&gradients_1/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"gradients_1/logistic_loss_grad/SumSum*gradients_1/adversalrial_loss_grad/truediv4gradients_1/logistic_loss_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&gradients_1/logistic_loss_grad/ReshapeReshape"gradients_1/logistic_loss_grad/Sum$gradients_1/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
$gradients_1/logistic_loss_grad/Sum_1Sum*gradients_1/adversalrial_loss_grad/truediv6gradients_1/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(gradients_1/logistic_loss_grad/Reshape_1Reshape$gradients_1/logistic_loss_grad/Sum_1&gradients_1/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/gradients_1/logistic_loss_grad/tuple/group_depsNoOp'^gradients_1/logistic_loss_grad/Reshape)^gradients_1/logistic_loss_grad/Reshape_1
�
7gradients_1/logistic_loss_grad/tuple/control_dependencyIdentity&gradients_1/logistic_loss_grad/Reshape0^gradients_1/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/logistic_loss_grad/Reshape*'
_output_shapes
:���������
�
9gradients_1/logistic_loss_grad/tuple/control_dependency_1Identity(gradients_1/logistic_loss_grad/Reshape_10^gradients_1/logistic_loss_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/logistic_loss_grad/Reshape_1*'
_output_shapes
:���������*
T0
`
gradients_1/Abs_grad/SignSignSub*
T0*/
_output_shapes
:���������
�
gradients_1/Abs_grad/mulMul'gradients_1/pixelwise_loss_grad/truedivgradients_1/Abs_grad/Sign*/
_output_shapes
:���������*
T0
|
(gradients_1/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
out_type0*
_output_shapes
:*
T0
{
*gradients_1/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
_output_shapes
:*
T0*
out_type0
�
8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/sub_grad/Shape*gradients_1/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients_1/logistic_loss/sub_grad/SumSum7gradients_1/logistic_loss_grad/tuple/control_dependency8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients_1/logistic_loss/sub_grad/ReshapeReshape&gradients_1/logistic_loss/sub_grad/Sum(gradients_1/logistic_loss/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
(gradients_1/logistic_loss/sub_grad/Sum_1Sum7gradients_1/logistic_loss_grad/tuple/control_dependency:gradients_1/logistic_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
z
&gradients_1/logistic_loss/sub_grad/NegNeg(gradients_1/logistic_loss/sub_grad/Sum_1*
_output_shapes
:*
T0
�
,gradients_1/logistic_loss/sub_grad/Reshape_1Reshape&gradients_1/logistic_loss/sub_grad/Neg*gradients_1/logistic_loss/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
3gradients_1/logistic_loss/sub_grad/tuple/group_depsNoOp+^gradients_1/logistic_loss/sub_grad/Reshape-^gradients_1/logistic_loss/sub_grad/Reshape_1
�
;gradients_1/logistic_loss/sub_grad/tuple/control_dependencyIdentity*gradients_1/logistic_loss/sub_grad/Reshape4^gradients_1/logistic_loss/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@gradients_1/logistic_loss/sub_grad/Reshape
�
=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1Identity,gradients_1/logistic_loss/sub_grad/Reshape_14^gradients_1/logistic_loss/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:���������
�
*gradients_1/logistic_loss/Log1p_grad/add/xConst:^gradients_1/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
(gradients_1/logistic_loss/Log1p_grad/addAdd*gradients_1/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:���������
�
/gradients_1/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_1/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:���������
�
(gradients_1/logistic_loss/Log1p_grad/mulMul9gradients_1/logistic_loss_grad/tuple/control_dependency_1/gradients_1/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:���������
o
gradients_1/Sub_grad/ShapeShapeDecoder/reshape_image*
out_type0*
_output_shapes
:*
T0
k
gradients_1/Sub_grad/Shape_1ShapeEncoder/real_in*
T0*
out_type0*
_output_shapes
:
�
*gradients_1/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Sub_grad/Shapegradients_1/Sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/Sub_grad/SumSumgradients_1/Abs_grad/mul*gradients_1/Sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients_1/Sub_grad/ReshapeReshapegradients_1/Sub_grad/Sumgradients_1/Sub_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
gradients_1/Sub_grad/Sum_1Sumgradients_1/Abs_grad/mul,gradients_1/Sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
^
gradients_1/Sub_grad/NegNeggradients_1/Sub_grad/Sum_1*
_output_shapes
:*
T0
�
gradients_1/Sub_grad/Reshape_1Reshapegradients_1/Sub_grad/Neggradients_1/Sub_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:���������
m
%gradients_1/Sub_grad/tuple/group_depsNoOp^gradients_1/Sub_grad/Reshape^gradients_1/Sub_grad/Reshape_1
�
-gradients_1/Sub_grad/tuple/control_dependencyIdentitygradients_1/Sub_grad/Reshape&^gradients_1/Sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/Sub_grad/Reshape*/
_output_shapes
:���������
�
/gradients_1/Sub_grad/tuple/control_dependency_1Identitygradients_1/Sub_grad/Reshape_1&^gradients_1/Sub_grad/tuple/group_deps*/
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients_1/Sub_grad/Reshape_1
�
0gradients_1/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:���������*
T0
�
,gradients_1/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual;gradients_1/logistic_loss/sub_grad/tuple/control_dependency0gradients_1/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:���������
�
.gradients_1/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_1/logistic_loss/Select_grad/zeros_like;gradients_1/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
6gradients_1/logistic_loss/Select_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss/Select_grad/Select/^gradients_1/logistic_loss/Select_grad/Select_1
�
>gradients_1/logistic_loss/Select_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss/Select_grad/Select7^gradients_1/logistic_loss/Select_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select*'
_output_shapes
:���������*
T0
�
@gradients_1/logistic_loss/Select_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss/Select_grad/Select_17^gradients_1/logistic_loss/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss/Select_grad/Select_1*'
_output_shapes
:���������
�
(gradients_1/logistic_loss/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
s
*gradients_1/logistic_loss/mul_grad/Shape_1Shape	ones_like*
T0*
out_type0*
_output_shapes
:
�
8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/mul_grad/Shape*gradients_1/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients_1/logistic_loss/mul_grad/MulMul=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*
T0*'
_output_shapes
:���������
�
&gradients_1/logistic_loss/mul_grad/SumSum&gradients_1/logistic_loss/mul_grad/Mul8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients_1/logistic_loss/mul_grad/ReshapeReshape&gradients_1/logistic_loss/mul_grad/Sum(gradients_1/logistic_loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients_1/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
(gradients_1/logistic_loss/mul_grad/Sum_1Sum(gradients_1/logistic_loss/mul_grad/Mul_1:gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
,gradients_1/logistic_loss/mul_grad/Reshape_1Reshape(gradients_1/logistic_loss/mul_grad/Sum_1*gradients_1/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
3gradients_1/logistic_loss/mul_grad/tuple/group_depsNoOp+^gradients_1/logistic_loss/mul_grad/Reshape-^gradients_1/logistic_loss/mul_grad/Reshape_1
�
;gradients_1/logistic_loss/mul_grad/tuple/control_dependencyIdentity*gradients_1/logistic_loss/mul_grad/Reshape4^gradients_1/logistic_loss/mul_grad/tuple/group_deps*=
_class3
1/loc:@gradients_1/logistic_loss/mul_grad/Reshape*'
_output_shapes
:���������*
T0
�
=gradients_1/logistic_loss/mul_grad/tuple/control_dependency_1Identity,gradients_1/logistic_loss/mul_grad/Reshape_14^gradients_1/logistic_loss/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:���������
�
&gradients_1/logistic_loss/Exp_grad/mulMul(gradients_1/logistic_loss/Log1p_grad/mullogistic_loss/Exp*'
_output_shapes
:���������*
T0
~
,gradients_1/Decoder/reshape_image_grad/ShapeShapeDecoder/last_layer*
T0*
out_type0*
_output_shapes
:
�
.gradients_1/Decoder/reshape_image_grad/ReshapeReshape-gradients_1/Sub_grad/tuple/control_dependency,gradients_1/Decoder/reshape_image_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
2gradients_1/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:���������
�
.gradients_1/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_1/logistic_loss/Exp_grad/mul2gradients_1/logistic_loss/Select_1_grad/zeros_like*'
_output_shapes
:���������*
T0
�
0gradients_1/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_1/logistic_loss/Select_1_grad/zeros_like&gradients_1/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:���������
�
8gradients_1/logistic_loss/Select_1_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss/Select_1_grad/Select1^gradients_1/logistic_loss/Select_1_grad/Select_1
�
@gradients_1/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss/Select_1_grad/Select9^gradients_1/logistic_loss/Select_1_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/logistic_loss/Select_1_grad/Select*'
_output_shapes
:���������*
T0
�
Bgradients_1/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss/Select_1_grad/Select_19^gradients_1/logistic_loss/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss/Select_1_grad/Select_1*'
_output_shapes
:���������
�
,gradients_1/Decoder/last_layer_grad/TanhGradTanhGradDecoder/last_layer.gradients_1/Decoder/reshape_image_grad/Reshape*
T0*(
_output_shapes
:����������
�
&gradients_1/logistic_loss/Neg_grad/NegNeg@gradients_1/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
2gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/Decoder/last_layer_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
7gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad-^gradients_1/Decoder/last_layer_grad/TanhGrad
�
?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_1/Decoder/last_layer_grad/TanhGrad8^gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/Decoder/last_layer_grad/TanhGrad*(
_output_shapes
:����������
�
Agradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad8^gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
,gradients_1/Decoder/dense/MatMul_grad/MatMulMatMul?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependencyDecoder/dense/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
.gradients_1/Decoder/dense/MatMul_grad/MatMul_1MatMulDecoder/second_layer/leaky_relu?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
�
6gradients_1/Decoder/dense/MatMul_grad/tuple/group_depsNoOp-^gradients_1/Decoder/dense/MatMul_grad/MatMul/^gradients_1/Decoder/dense/MatMul_grad/MatMul_1
�
>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/Decoder/dense/MatMul_grad/MatMul7^gradients_1/Decoder/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/Decoder/dense/MatMul_grad/MatMul*(
_output_shapes
:����������
�
@gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/Decoder/dense/MatMul_grad/MatMul_17^gradients_1/Decoder/dense/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/Decoder/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
gradients_1/AddNAddN>gradients_1/logistic_loss/Select_grad/tuple/control_dependency;gradients_1/logistic_loss/mul_grad/tuple/control_dependencyBgradients_1/logistic_loss/Select_1_grad/tuple/control_dependency_1&gradients_1/logistic_loss/Neg_grad/Neg*
N*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select
�
7gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes
:
�
<gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN8^gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad
�
Dgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN=^gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select*'
_output_shapes
:���������*
T0
�
Fgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity7gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad=^gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
6gradients_1/Decoder/second_layer/leaky_relu_grad/ShapeShape#Decoder/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1Shape8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_2Shape>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
<gradients_1/Decoder/second_layer/leaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
6gradients_1/Decoder/second_layer/leaky_relu_grad/zerosFill8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_2<gradients_1/Decoder/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Decoder/second_layer/leaky_relu/mul8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Fgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Decoder/second_layer/leaky_relu_grad/Shape8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients_1/Decoder/second_layer/leaky_relu_grad/SelectSelect=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqual>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency6gradients_1/Decoder/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
9gradients_1/Decoder/second_layer/leaky_relu_grad/Select_1Select=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqual6gradients_1/Decoder/second_layer/leaky_relu_grad/zeros>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
4gradients_1/Decoder/second_layer/leaky_relu_grad/SumSum7gradients_1/Decoder/second_layer/leaky_relu_grad/SelectFgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients_1/Decoder/second_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Decoder/second_layer/leaky_relu_grad/Sum6gradients_1/Decoder/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
6gradients_1/Decoder/second_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Decoder/second_layer/leaky_relu_grad/Select_1Hgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Decoder/second_layer/leaky_relu_grad/Sum_18gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Agradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape;^gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1
�
Igradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Decoder/second_layer/leaky_relu_grad/ReshapeB^gradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*K
_classA
?=loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape
�
Kgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1B^gradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
1gradients_1/Discriminator/prob/MatMul_grad/MatMulMatMulDgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
3gradients_1/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluDgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
;gradients_1/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp2^gradients_1/Discriminator/prob/MatMul_grad/MatMul4^gradients_1/Discriminator/prob/MatMul_grad/MatMul_1
�
Cgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity1gradients_1/Discriminator/prob/MatMul_grad/MatMul<^gradients_1/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/Discriminator/prob/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Egradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity3gradients_1/Discriminator/prob/MatMul_grad/MatMul_1<^gradients_1/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	�
}
:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1Shape8Decoder/second_layer/batch_normalization/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
�
Jgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/MulMulIgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/SumSum8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/MulJgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Mul_1Mul%Decoder/second_layer/leaky_relu/alphaIgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Egradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1
�
Mgradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*O
_classE
CAloc:@gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
�
Ogradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
<gradients_1/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
�
>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Bgradients_1/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
<gradients_1/Discriminator/second_layer/leaky_relu_grad/zerosFill>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_2Bgradients_1/Discriminator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Cgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Lgradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
=gradients_1/Discriminator/second_layer/leaky_relu_grad/SelectSelectCgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqualCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency<gradients_1/Discriminator/second_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
?gradients_1/Discriminator/second_layer/leaky_relu_grad/Select_1SelectCgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqual<gradients_1/Discriminator/second_layer/leaky_relu_grad/zerosCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
:gradients_1/Discriminator/second_layer/leaky_relu_grad/SumSum=gradients_1/Discriminator/second_layer/leaky_relu_grad/SelectLgradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape:gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum<gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
<gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum?gradients_1/Discriminator/second_layer/leaky_relu_grad/Select_1Ngradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape<gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum_1>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Ggradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp?^gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeA^gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1
�
Ogradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity>gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeH^gradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Qgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1H^gradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_1AddNKgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Decoder/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_1_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_1agradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Sgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Pgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ShapeBgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/MulMulOgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/SumSum>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/MulPgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaOgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Rgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Dgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Kgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
�
Sgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeL^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ugradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1L^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Decoder/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Decoder/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
�
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Decoder/second_layer/fully_connected/BiasAddbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
Kgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
Xgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg
�
`gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
gradients_1/AddN_2AddNQgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Ugradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Tgradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2P^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2U^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradU^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Igradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ngradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Vgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
Xgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Decoder/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:�*
T0
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Decoder/second_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:�
�
Zgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*`
_classV
TRloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul
�
dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Igradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMul\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Kgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_relu\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Sgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulL^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
�
[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulT^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
]gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1T^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
Cgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Decoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Egradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1MatMulDecoder/first_layer/leaky_reluVgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Mgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1
�
Ugradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Wgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_3AddNdgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Kgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_33Decoder/second_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
�
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_38Decoder/second_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:�*
T0
�
Xgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
`gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*^
_classT
RPloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul
�
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
;gradients_1/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_2Shape[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Agradients_1/Discriminator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;gradients_1/Discriminator/first_layer/leaky_relu_grad/zerosFill=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_2Agradients_1/Discriminator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Bgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Kgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
<gradients_1/Discriminator/first_layer/leaky_relu_grad/SelectSelectBgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqual[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency;gradients_1/Discriminator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
>gradients_1/Discriminator/first_layer/leaky_relu_grad/Select_1SelectBgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqual;gradients_1/Discriminator/first_layer/leaky_relu_grad/zeros[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
9gradients_1/Discriminator/first_layer/leaky_relu_grad/SumSum<gradients_1/Discriminator/first_layer/leaky_relu_grad/SelectKgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
=gradients_1/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape9gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum;gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
;gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum>gradients_1/Discriminator/first_layer/leaky_relu_grad/Select_1Mgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape;gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum_1=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Fgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp>^gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape@^gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1
�
Ngradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity=gradients_1/Discriminator/first_layer/leaky_relu_grad/ReshapeG^gradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*P
_classF
DBloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape
�
Pgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity?gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1G^gradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1
�
5gradients_1/Decoder/first_layer/leaky_relu_grad/ShapeShape"Decoder/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1Shape+Decoder/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
;gradients_1/Decoder/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
5gradients_1/Decoder/first_layer/leaky_relu_grad/zerosFill7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_2;gradients_1/Decoder/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual"Decoder/first_layer/leaky_relu/mul+Decoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Egradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/Decoder/first_layer/leaky_relu_grad/Shape7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6gradients_1/Decoder/first_layer/leaky_relu_grad/SelectSelect<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqualUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency5gradients_1/Decoder/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
8gradients_1/Decoder/first_layer/leaky_relu_grad/Select_1Select<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqual5gradients_1/Decoder/first_layer/leaky_relu_grad/zerosUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
3gradients_1/Decoder/first_layer/leaky_relu_grad/SumSum6gradients_1/Decoder/first_layer/leaky_relu_grad/SelectEgradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients_1/Decoder/first_layer/leaky_relu_grad/ReshapeReshape3gradients_1/Decoder/first_layer/leaky_relu_grad/Sum5gradients_1/Decoder/first_layer/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
5gradients_1/Decoder/first_layer/leaky_relu_grad/Sum_1Sum8gradients_1/Decoder/first_layer/leaky_relu_grad/Select_1Ggradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
9gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1Reshape5gradients_1/Decoder/first_layer/leaky_relu_grad/Sum_17gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
@gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_depsNoOp8^gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape:^gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1
�
Hgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity7gradients_1/Decoder/first_layer/leaky_relu_grad/ReshapeA^gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Jgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity9gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1A^gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Ogradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ShapeAgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/MulMulNgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/SumSum=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/MulOgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaNgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Qgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Cgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Jgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeD^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
�
Rgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeK^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Tgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1K^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*V
_classL
JHloc:@gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
|
9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1Shape+Decoder/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Igradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/MulMulHgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency+Decoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/SumSum7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/MulIgradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ReshapeReshape7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Mul_1Mul$Decoder/first_layer/leaky_relu/alphaHgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum_1Sum9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Mul_1Kgradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1Reshape9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum_1;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Dgradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp<^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape>^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1
�
Lgradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ngradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity=gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1E^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*P
_classF
DBloc:@gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1
�
gradients_1/AddN_4AddNPgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Tgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ngradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_4*
_output_shapes	
:�*
T0*
data_formatNHWC
�
Sgradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_4O^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_4T^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
]gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradT^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
gradients_1/AddN_5AddNJgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Ngradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1*
N
�
Hgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Mgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_5I^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ugradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_5N^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Wgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradN^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Hgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMul[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
�
Jgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_code[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d�*
transpose_a(*
transpose_b( *
T0
�
Rgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulK^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Zgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulS^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d*
T0
�
\gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1S^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
_output_shapes
:	d�*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Bgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulMatMulUgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency/Decoder/first_layer/fully_connected/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(
�
Dgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_codeUgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d�*
transpose_a(*
transpose_b( 
�
Lgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpC^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulE^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Tgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityBgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulM^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
Vgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityDgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1M^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d�
�
gradients_1/AddN_6AddNZgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyTgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*[
_classQ
OMloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*
N*'
_output_shapes
:���������d
�
1gradients_1/Encoder/encoder_code_grad/SigmoidGradSigmoidGradEncoder/encoder_codegradients_1/AddN_6*
T0*'
_output_shapes
:���������d
t
"gradients_1/Encoder/Add_grad/ShapeShapeEncoder/logvar_std*
_output_shapes
:*
T0*
out_type0
~
$gradients_1/Encoder/Add_grad/Shape_1ShapeEncoder/encoder_mu/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
2gradients_1/Encoder/Add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_1/Encoder/Add_grad/Shape$gradients_1/Encoder/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients_1/Encoder/Add_grad/SumSum1gradients_1/Encoder/encoder_code_grad/SigmoidGrad2gradients_1/Encoder/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
$gradients_1/Encoder/Add_grad/ReshapeReshape gradients_1/Encoder/Add_grad/Sum"gradients_1/Encoder/Add_grad/Shape*'
_output_shapes
:���������d*
T0*
Tshape0
�
"gradients_1/Encoder/Add_grad/Sum_1Sum1gradients_1/Encoder/encoder_code_grad/SigmoidGrad4gradients_1/Encoder/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&gradients_1/Encoder/Add_grad/Reshape_1Reshape"gradients_1/Encoder/Add_grad/Sum_1$gradients_1/Encoder/Add_grad/Shape_1*
Tshape0*'
_output_shapes
:���������d*
T0
�
-gradients_1/Encoder/Add_grad/tuple/group_depsNoOp%^gradients_1/Encoder/Add_grad/Reshape'^gradients_1/Encoder/Add_grad/Reshape_1
�
5gradients_1/Encoder/Add_grad/tuple/control_dependencyIdentity$gradients_1/Encoder/Add_grad/Reshape.^gradients_1/Encoder/Add_grad/tuple/group_deps*7
_class-
+)loc:@gradients_1/Encoder/Add_grad/Reshape*'
_output_shapes
:���������d*
T0
�
7gradients_1/Encoder/Add_grad/tuple/control_dependency_1Identity&gradients_1/Encoder/Add_grad/Reshape_1.^gradients_1/Encoder/Add_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*9
_class/
-+loc:@gradients_1/Encoder/Add_grad/Reshape_1
s
)gradients_1/Encoder/logvar_std_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:d
v
+gradients_1/Encoder/logvar_std_grad/Shape_1ShapeEncoder/Exp*
T0*
out_type0*
_output_shapes
:
�
9gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients_1/Encoder/logvar_std_grad/Shape+gradients_1/Encoder/logvar_std_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
'gradients_1/Encoder/logvar_std_grad/MulMul5gradients_1/Encoder/Add_grad/tuple/control_dependencyEncoder/Exp*
T0*'
_output_shapes
:���������d
�
'gradients_1/Encoder/logvar_std_grad/SumSum'gradients_1/Encoder/logvar_std_grad/Mul9gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
+gradients_1/Encoder/logvar_std_grad/ReshapeReshape'gradients_1/Encoder/logvar_std_grad/Sum)gradients_1/Encoder/logvar_std_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
)gradients_1/Encoder/logvar_std_grad/Mul_1MulEncoder/random_normal5gradients_1/Encoder/Add_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������d
�
)gradients_1/Encoder/logvar_std_grad/Sum_1Sum)gradients_1/Encoder/logvar_std_grad/Mul_1;gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
-gradients_1/Encoder/logvar_std_grad/Reshape_1Reshape)gradients_1/Encoder/logvar_std_grad/Sum_1+gradients_1/Encoder/logvar_std_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������d
�
4gradients_1/Encoder/logvar_std_grad/tuple/group_depsNoOp,^gradients_1/Encoder/logvar_std_grad/Reshape.^gradients_1/Encoder/logvar_std_grad/Reshape_1
�
<gradients_1/Encoder/logvar_std_grad/tuple/control_dependencyIdentity+gradients_1/Encoder/logvar_std_grad/Reshape5^gradients_1/Encoder/logvar_std_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_1/Encoder/logvar_std_grad/Reshape*
_output_shapes
:d
�
>gradients_1/Encoder/logvar_std_grad/tuple/control_dependency_1Identity-gradients_1/Encoder/logvar_std_grad/Reshape_15^gradients_1/Encoder/logvar_std_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*@
_class6
42loc:@gradients_1/Encoder/logvar_std_grad/Reshape_1
�
7gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/Encoder/Add_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:d
�
<gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_depsNoOp8^gradients_1/Encoder/Add_grad/tuple/control_dependency_18^gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad
�
Dgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/Encoder/Add_grad/tuple/control_dependency_1=^gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/Encoder/Add_grad/Reshape_1*'
_output_shapes
:���������d
�
Fgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency_1Identity7gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad=^gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d*
T0
�
 gradients_1/Encoder/Exp_grad/mulMul>gradients_1/Encoder/logvar_std_grad/tuple/control_dependency_1Encoder/Exp*
T0*'
_output_shapes
:���������d
�
1gradients_1/Encoder/encoder_mu/MatMul_grad/MatMulMatMulDgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependencyEncoder/encoder_mu/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
3gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1MatMulEncoder/second_layer/leaky_reluDgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	�d*
transpose_a(*
transpose_b( *
T0
�
;gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_depsNoOp2^gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul4^gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1
�
Cgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependencyIdentity1gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul<^gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Egradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependency_1Identity3gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1<^gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1*
_output_shapes
:	�d
�
&gradients_1/Encoder/truediv_grad/ShapeShapeEncoder/encoder_logvar/BiasAdd*
T0*
out_type0*
_output_shapes
:
k
(gradients_1/Encoder/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
6gradients_1/Encoder/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/Encoder/truediv_grad/Shape(gradients_1/Encoder/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
(gradients_1/Encoder/truediv_grad/RealDivRealDiv gradients_1/Encoder/Exp_grad/mulEncoder/truediv/y*'
_output_shapes
:���������d*
T0
�
$gradients_1/Encoder/truediv_grad/SumSum(gradients_1/Encoder/truediv_grad/RealDiv6gradients_1/Encoder/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients_1/Encoder/truediv_grad/ReshapeReshape$gradients_1/Encoder/truediv_grad/Sum&gradients_1/Encoder/truediv_grad/Shape*
Tshape0*'
_output_shapes
:���������d*
T0
}
$gradients_1/Encoder/truediv_grad/NegNegEncoder/encoder_logvar/BiasAdd*
T0*'
_output_shapes
:���������d
�
*gradients_1/Encoder/truediv_grad/RealDiv_1RealDiv$gradients_1/Encoder/truediv_grad/NegEncoder/truediv/y*'
_output_shapes
:���������d*
T0
�
*gradients_1/Encoder/truediv_grad/RealDiv_2RealDiv*gradients_1/Encoder/truediv_grad/RealDiv_1Encoder/truediv/y*'
_output_shapes
:���������d*
T0
�
$gradients_1/Encoder/truediv_grad/mulMul gradients_1/Encoder/Exp_grad/mul*gradients_1/Encoder/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:���������d
�
&gradients_1/Encoder/truediv_grad/Sum_1Sum$gradients_1/Encoder/truediv_grad/mul8gradients_1/Encoder/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients_1/Encoder/truediv_grad/Reshape_1Reshape&gradients_1/Encoder/truediv_grad/Sum_1(gradients_1/Encoder/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
1gradients_1/Encoder/truediv_grad/tuple/group_depsNoOp)^gradients_1/Encoder/truediv_grad/Reshape+^gradients_1/Encoder/truediv_grad/Reshape_1
�
9gradients_1/Encoder/truediv_grad/tuple/control_dependencyIdentity(gradients_1/Encoder/truediv_grad/Reshape2^gradients_1/Encoder/truediv_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*;
_class1
/-loc:@gradients_1/Encoder/truediv_grad/Reshape
�
;gradients_1/Encoder/truediv_grad/tuple/control_dependency_1Identity*gradients_1/Encoder/truediv_grad/Reshape_12^gradients_1/Encoder/truediv_grad/tuple/group_deps*
_output_shapes
: *
T0*=
_class3
1/loc:@gradients_1/Encoder/truediv_grad/Reshape_1
�
;gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients_1/Encoder/truediv_grad/tuple/control_dependency*
_output_shapes
:d*
T0*
data_formatNHWC
�
@gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_depsNoOp<^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGrad:^gradients_1/Encoder/truediv_grad/tuple/control_dependency
�
Hgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependencyIdentity9gradients_1/Encoder/truediv_grad/tuple/control_dependencyA^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/Encoder/truediv_grad/Reshape*'
_output_shapes
:���������d
�
Jgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGradA^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_deps*
_output_shapes
:d*
T0*N
_classD
B@loc:@gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGrad
�
5gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMulMatMulHgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency"Encoder/encoder_logvar/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
7gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1MatMulEncoder/second_layer/leaky_reluHgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�d*
transpose_a(*
transpose_b( 
�
?gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_depsNoOp6^gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul8^gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1
�
Ggradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependencyIdentity5gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul@^gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Igradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency_1Identity7gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1@^gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1*
_output_shapes
:	�d
�
gradients_1/AddN_7AddNCgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependencyGgradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency*
T0*D
_class:
86loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul*
N*(
_output_shapes
:����������
�
6gradients_1/Encoder/second_layer/leaky_relu_grad/ShapeShape#Encoder/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1Shape8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_2Shapegradients_1/AddN_7*
T0*
out_type0*
_output_shapes
:
�
<gradients_1/Encoder/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6gradients_1/Encoder/second_layer/leaky_relu_grad/zerosFill8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_2<gradients_1/Encoder/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Encoder/second_layer/leaky_relu/mul8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Fgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Encoder/second_layer/leaky_relu_grad/Shape8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients_1/Encoder/second_layer/leaky_relu_grad/SelectSelect=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqualgradients_1/AddN_76gradients_1/Encoder/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
9gradients_1/Encoder/second_layer/leaky_relu_grad/Select_1Select=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqual6gradients_1/Encoder/second_layer/leaky_relu_grad/zerosgradients_1/AddN_7*
T0*(
_output_shapes
:����������
�
4gradients_1/Encoder/second_layer/leaky_relu_grad/SumSum7gradients_1/Encoder/second_layer/leaky_relu_grad/SelectFgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
8gradients_1/Encoder/second_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Encoder/second_layer/leaky_relu_grad/Sum6gradients_1/Encoder/second_layer/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
6gradients_1/Encoder/second_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Encoder/second_layer/leaky_relu_grad/Select_1Hgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
:gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Encoder/second_layer/leaky_relu_grad/Sum_18gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Agradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape;^gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1
�
Igradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Encoder/second_layer/leaky_relu_grad/ReshapeB^gradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Kgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1B^gradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
}
:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1Shape8Encoder/second_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
Jgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/MulMulIgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency8Encoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/SumSum8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/MulJgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Mul_1Mul%Encoder/second_layer/leaky_relu/alphaIgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Egradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1
�
Mgradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ogradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*Q
_classG
ECloc:@gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1
�
gradients_1/AddN_8AddNKgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Encoder/second_layer/batch_normalization/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
�
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_8_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_8agradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Sgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Encoder/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Encoder/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Encoder/second_layer/fully_connected/BiasAddbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Sgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Kgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
Xgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg
�
`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Igradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
_output_shapes	
:�*
T0*
data_formatNHWC
�
Ngradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Vgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
Xgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Encoder/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Encoder/second_layer/batch_normalization/moving_mean/read*
_output_shapes	
:�*
T0
�
Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*`
_classV
TRloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul
�
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Cgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Encoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Egradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/first_layer/leaky_reluVgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Mgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1
�
Ugradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*V
_classL
JHloc:@gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul
�
Wgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*X
_classN
LJloc:@gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1
�
gradients_1/AddN_9AddNdgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Kgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_93Encoder/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_98Encoder/second_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
Xgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*`
_classV
TRloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
5gradients_1/Encoder/first_layer/leaky_relu_grad/ShapeShape"Encoder/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1Shape+Encoder/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
;gradients_1/Encoder/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
5gradients_1/Encoder/first_layer/leaky_relu_grad/zerosFill7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_2;gradients_1/Encoder/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual"Encoder/first_layer/leaky_relu/mul+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Egradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/Encoder/first_layer/leaky_relu_grad/Shape7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6gradients_1/Encoder/first_layer/leaky_relu_grad/SelectSelect<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqualUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency5gradients_1/Encoder/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
8gradients_1/Encoder/first_layer/leaky_relu_grad/Select_1Select<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqual5gradients_1/Encoder/first_layer/leaky_relu_grad/zerosUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
3gradients_1/Encoder/first_layer/leaky_relu_grad/SumSum6gradients_1/Encoder/first_layer/leaky_relu_grad/SelectEgradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
7gradients_1/Encoder/first_layer/leaky_relu_grad/ReshapeReshape3gradients_1/Encoder/first_layer/leaky_relu_grad/Sum5gradients_1/Encoder/first_layer/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
5gradients_1/Encoder/first_layer/leaky_relu_grad/Sum_1Sum8gradients_1/Encoder/first_layer/leaky_relu_grad/Select_1Ggradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
9gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1Reshape5gradients_1/Encoder/first_layer/leaky_relu_grad/Sum_17gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
@gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_depsNoOp8^gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape:^gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1
�
Hgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity7gradients_1/Encoder/first_layer/leaky_relu_grad/ReshapeA^gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*J
_class@
><loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape
�
Jgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity9gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1A^gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
|
9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1Shape+Encoder/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Igradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/MulMulHgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency+Encoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/SumSum7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/MulIgradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ReshapeReshape7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Mul_1Mul$Encoder/first_layer/leaky_relu/alphaHgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum_1Sum9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Mul_1Kgradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1Reshape9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum_1;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Dgradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp<^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape>^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1
�
Lgradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ngradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity=gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1E^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_10AddNJgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Ngradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Hgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_10*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Mgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_10I^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ugradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_10N^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1
�
Wgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradN^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Bgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulMatMulUgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency/Encoder/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Dgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/ReshapeUgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Lgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpC^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulE^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Tgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityBgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulM^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Vgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityDgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1M^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
beta1_power_1/initial_valueConst*%
_class
loc:@Decoder/dense/bias*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
beta1_power_1
VariableV2*
shared_name *%
_class
loc:@Decoder/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes
: 
u
beta1_power_1/readIdentitybeta1_power_1*
_output_shapes
: *
T0*%
_class
loc:@Decoder/dense/bias
�
beta2_power_1/initial_valueConst*
dtype0*
_output_shapes
: *%
_class
loc:@Decoder/dense/bias*
valueB
 *w�?
�
beta2_power_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@Decoder/dense/bias*
	container 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(
u
beta2_power_1/readIdentitybeta2_power_1*
_output_shapes
: *
T0*%
_class
loc:@Decoder/dense/bias
�
[Encoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
QEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
KEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill[Encoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorQEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
9Encoder/first_layer/fully_connected/kernel/generator_opti
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container *
shape:
��
�
@Encoder/first_layer/fully_connected/kernel/generator_opti/AssignAssign9Encoder/first_layer/fully_connected/kernel/generator_optiKEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
>Encoder/first_layer/fully_connected/kernel/generator_opti/readIdentity9Encoder/first_layer/fully_connected/kernel/generator_opti*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0
�
]Encoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
SEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*
_output_shapes
: *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
�
MEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill]Encoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorSEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*

index_type0
�
;Encoder/first_layer/fully_connected/kernel/generator_opti_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container *
shape:
��
�
BEncoder/first_layer/fully_connected/kernel/generator_opti_1/AssignAssign;Encoder/first_layer/fully_connected/kernel/generator_opti_1MEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
@Encoder/first_layer/fully_connected/kernel/generator_opti_1/readIdentity;Encoder/first_layer/fully_connected/kernel/generator_opti_1*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
IEncoder/first_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7Encoder/first_layer/fully_connected/bias/generator_opti
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
	container *
shape:�
�
>Encoder/first_layer/fully_connected/bias/generator_opti/AssignAssign7Encoder/first_layer/fully_connected/bias/generator_optiIEncoder/first_layer/fully_connected/bias/generator_opti/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(
�
<Encoder/first_layer/fully_connected/bias/generator_opti/readIdentity7Encoder/first_layer/fully_connected/bias/generator_opti*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:�
�
KEncoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
9Encoder/first_layer/fully_connected/bias/generator_opti_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
	container 
�
@Encoder/first_layer/fully_connected/bias/generator_opti_1/AssignAssign9Encoder/first_layer/fully_connected/bias/generator_opti_1KEncoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
>Encoder/first_layer/fully_connected/bias/generator_opti_1/readIdentity9Encoder/first_layer/fully_connected/bias/generator_opti_1*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:�
�
\Encoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
REncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LEncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill\Encoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorREncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*

index_type0
�
:Encoder/second_layer/fully_connected/kernel/generator_opti
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:
��
�
AEncoder/second_layer/fully_connected/kernel/generator_opti/AssignAssign:Encoder/second_layer/fully_connected/kernel/generator_optiLEncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
?Encoder/second_layer/fully_connected/kernel/generator_opti/readIdentity:Encoder/second_layer/fully_connected/kernel/generator_opti*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0
�
^Encoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
TEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill^Encoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorTEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*

index_type0
�
<Encoder/second_layer/fully_connected/kernel/generator_opti_1
VariableV2*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
CEncoder/second_layer/fully_connected/kernel/generator_opti_1/AssignAssign<Encoder/second_layer/fully_connected/kernel/generator_opti_1NEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
AEncoder/second_layer/fully_connected/kernel/generator_opti_1/readIdentity<Encoder/second_layer/fully_connected/kernel/generator_opti_1*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
JEncoder/second_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
8Encoder/second_layer/fully_connected/bias/generator_opti
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container *
shape:�
�
?Encoder/second_layer/fully_connected/bias/generator_opti/AssignAssign8Encoder/second_layer/fully_connected/bias/generator_optiJEncoder/second_layer/fully_connected/bias/generator_opti/Initializer/zeros*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
=Encoder/second_layer/fully_connected/bias/generator_opti/readIdentity8Encoder/second_layer/fully_connected/bias/generator_opti*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
_output_shapes	
:�*
T0
�
LEncoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
:Encoder/second_layer/fully_connected/bias/generator_opti_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container *
shape:�
�
AEncoder/second_layer/fully_connected/bias/generator_opti_1/AssignAssign:Encoder/second_layer/fully_connected/bias/generator_opti_1LEncoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
?Encoder/second_layer/fully_connected/bias/generator_opti_1/readIdentity:Encoder/second_layer/fully_connected/bias/generator_opti_1*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
_output_shapes	
:�
�
OEncoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB�*    *
dtype0
�
=Encoder/second_layer/batch_normalization/gamma/generator_opti
VariableV2*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
DEncoder/second_layer/batch_normalization/gamma/generator_opti/AssignAssign=Encoder/second_layer/batch_normalization/gamma/generator_optiOEncoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
BEncoder/second_layer/batch_normalization/gamma/generator_opti/readIdentity=Encoder/second_layer/batch_normalization/gamma/generator_opti*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
QEncoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zerosConst*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
?Encoder/second_layer/batch_normalization/gamma/generator_opti_1
VariableV2*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0
�
FEncoder/second_layer/batch_normalization/gamma/generator_opti_1/AssignAssign?Encoder/second_layer/batch_normalization/gamma/generator_opti_1QEncoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(
�
DEncoder/second_layer/batch_normalization/gamma/generator_opti_1/readIdentity?Encoder/second_layer/batch_normalization/gamma/generator_opti_1*
_output_shapes	
:�*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma
�
NEncoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB�*    
�
<Encoder/second_layer/batch_normalization/beta/generator_opti
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
	container 
�
CEncoder/second_layer/batch_normalization/beta/generator_opti/AssignAssign<Encoder/second_layer/batch_normalization/beta/generator_optiNEncoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(
�
AEncoder/second_layer/batch_normalization/beta/generator_opti/readIdentity<Encoder/second_layer/batch_normalization/beta/generator_opti*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
PEncoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zerosConst*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
>Encoder/second_layer/batch_normalization/beta/generator_opti_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
	container *
shape:�
�
EEncoder/second_layer/batch_normalization/beta/generator_opti_1/AssignAssign>Encoder/second_layer/batch_normalization/beta/generator_opti_1PEncoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
CEncoder/second_layer/batch_normalization/beta/generator_opti_1/readIdentity>Encoder/second_layer/batch_normalization/beta/generator_opti_1*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:�*
T0
�
JEncoder/encoder_mu/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0
�
@Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros/ConstConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
:Encoder/encoder_mu/kernel/generator_opti/Initializer/zerosFillJEncoder/encoder_mu/kernel/generator_opti/Initializer/zeros/shape_as_tensor@Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros/Const*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*

index_type0*
_output_shapes
:	�d
�
(Encoder/encoder_mu/kernel/generator_opti
VariableV2*
dtype0*
_output_shapes
:	�d*
shared_name *,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container *
shape:	�d
�
/Encoder/encoder_mu/kernel/generator_opti/AssignAssign(Encoder/encoder_mu/kernel/generator_opti:Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	�d
�
-Encoder/encoder_mu/kernel/generator_opti/readIdentity(Encoder/encoder_mu/kernel/generator_opti*
_output_shapes
:	�d*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel
�
LEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
�
BEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/ConstConst*
_output_shapes
: *,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *    *
dtype0
�
<Encoder/encoder_mu/kernel/generator_opti_1/Initializer/zerosFillLEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorBEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/Const*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*

index_type0*
_output_shapes
:	�d
�
*Encoder/encoder_mu/kernel/generator_opti_1
VariableV2*
dtype0*
_output_shapes
:	�d*
shared_name *,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container *
shape:	�d
�
1Encoder/encoder_mu/kernel/generator_opti_1/AssignAssign*Encoder/encoder_mu/kernel/generator_opti_1<Encoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	�d
�
/Encoder/encoder_mu/kernel/generator_opti_1/readIdentity*Encoder/encoder_mu/kernel/generator_opti_1*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	�d
�
8Encoder/encoder_mu/bias/generator_opti/Initializer/zerosConst**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0*
_output_shapes
:d
�
&Encoder/encoder_mu/bias/generator_opti
VariableV2*
shared_name **
_class 
loc:@Encoder/encoder_mu/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
-Encoder/encoder_mu/bias/generator_opti/AssignAssign&Encoder/encoder_mu/bias/generator_opti8Encoder/encoder_mu/bias/generator_opti/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
validate_shape(*
_output_shapes
:d
�
+Encoder/encoder_mu/bias/generator_opti/readIdentity&Encoder/encoder_mu/bias/generator_opti*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d
�
:Encoder/encoder_mu/bias/generator_opti_1/Initializer/zerosConst**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0*
_output_shapes
:d
�
(Encoder/encoder_mu/bias/generator_opti_1
VariableV2*
dtype0*
_output_shapes
:d*
shared_name **
_class 
loc:@Encoder/encoder_mu/bias*
	container *
shape:d
�
/Encoder/encoder_mu/bias/generator_opti_1/AssignAssign(Encoder/encoder_mu/bias/generator_opti_1:Encoder/encoder_mu/bias/generator_opti_1/Initializer/zeros*
_output_shapes
:d*
use_locking(*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
validate_shape(
�
-Encoder/encoder_mu/bias/generator_opti_1/readIdentity(Encoder/encoder_mu/bias/generator_opti_1*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d
�
NEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
�
DEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/ConstConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
>Encoder/encoder_logvar/kernel/generator_opti/Initializer/zerosFillNEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/shape_as_tensorDEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/Const*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*

index_type0*
_output_shapes
:	�d*
T0
�
,Encoder/encoder_logvar/kernel/generator_opti
VariableV2*
dtype0*
_output_shapes
:	�d*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	�d
�
3Encoder/encoder_logvar/kernel/generator_opti/AssignAssign,Encoder/encoder_logvar/kernel/generator_opti>Encoder/encoder_logvar/kernel/generator_opti/Initializer/zeros*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking(
�
1Encoder/encoder_logvar/kernel/generator_opti/readIdentity,Encoder/encoder_logvar/kernel/generator_opti*
_output_shapes
:	�d*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel
�
PEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
�
FEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/ConstConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@Encoder/encoder_logvar/kernel/generator_opti_1/Initializer/zerosFillPEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorFEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/Const*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*

index_type0*
_output_shapes
:	�d
�
.Encoder/encoder_logvar/kernel/generator_opti_1
VariableV2*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel
�
5Encoder/encoder_logvar/kernel/generator_opti_1/AssignAssign.Encoder/encoder_logvar/kernel/generator_opti_1@Encoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros*
_output_shapes
:	�d*
use_locking(*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(
�
3Encoder/encoder_logvar/kernel/generator_opti_1/readIdentity.Encoder/encoder_logvar/kernel/generator_opti_1*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	�d
�
<Encoder/encoder_logvar/bias/generator_opti/Initializer/zerosConst*
dtype0*
_output_shapes
:d*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    
�
*Encoder/encoder_logvar/bias/generator_opti
VariableV2*.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
�
1Encoder/encoder_logvar/bias/generator_opti/AssignAssign*Encoder/encoder_logvar/bias/generator_opti<Encoder/encoder_logvar/bias/generator_opti/Initializer/zeros*
_output_shapes
:d*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(
�
/Encoder/encoder_logvar/bias/generator_opti/readIdentity*Encoder/encoder_logvar/bias/generator_opti*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
:d
�
>Encoder/encoder_logvar/bias/generator_opti_1/Initializer/zerosConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0*
_output_shapes
:d
�
,Encoder/encoder_logvar/bias/generator_opti_1
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape:d
�
3Encoder/encoder_logvar/bias/generator_opti_1/AssignAssign,Encoder/encoder_logvar/bias/generator_opti_1>Encoder/encoder_logvar/bias/generator_opti_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
:d
�
1Encoder/encoder_logvar/bias/generator_opti_1/readIdentity,Encoder/encoder_logvar/bias/generator_opti_1*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
:d
�
[Decoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
�
QDecoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
KDecoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill[Decoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorQDecoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
9Decoder/first_layer/fully_connected/kernel/generator_opti
VariableV2*
shared_name *=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�
�
@Decoder/first_layer/fully_connected/kernel/generator_opti/AssignAssign9Decoder/first_layer/fully_connected/kernel/generator_optiKDecoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
_output_shapes
:	d�*
use_locking(*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
validate_shape(
�
>Decoder/first_layer/fully_connected/kernel/generator_opti/readIdentity9Decoder/first_layer/fully_connected/kernel/generator_opti*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
]Decoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
�
SDecoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*
_output_shapes
: *=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
�
MDecoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill]Decoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorSDecoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const*
_output_shapes
:	d�*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*

index_type0
�
;Decoder/first_layer/fully_connected/kernel/generator_opti_1
VariableV2*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name *=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel
�
BDecoder/first_layer/fully_connected/kernel/generator_opti_1/AssignAssign;Decoder/first_layer/fully_connected/kernel/generator_opti_1MDecoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel
�
@Decoder/first_layer/fully_connected/kernel/generator_opti_1/readIdentity;Decoder/first_layer/fully_connected/kernel/generator_opti_1*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
IDecoder/first_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7Decoder/first_layer/fully_connected/bias/generator_opti
VariableV2*
shared_name *;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
>Decoder/first_layer/fully_connected/bias/generator_opti/AssignAssign7Decoder/first_layer/fully_connected/bias/generator_optiIDecoder/first_layer/fully_connected/bias/generator_opti/Initializer/zeros*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
<Decoder/first_layer/fully_connected/bias/generator_opti/readIdentity7Decoder/first_layer/fully_connected/bias/generator_opti*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
_output_shapes	
:�*
T0
�
KDecoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
9Decoder/first_layer/fully_connected/bias/generator_opti_1
VariableV2*
shared_name *;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
@Decoder/first_layer/fully_connected/bias/generator_opti_1/AssignAssign9Decoder/first_layer/fully_connected/bias/generator_opti_1KDecoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
>Decoder/first_layer/fully_connected/bias/generator_opti_1/readIdentity9Decoder/first_layer/fully_connected/bias/generator_opti_1*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
_output_shapes	
:�
�
\Decoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0
�
RDecoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LDecoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill\Decoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorRDecoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
:Decoder/second_layer/fully_connected/kernel/generator_opti
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
	container *
shape:
��
�
ADecoder/second_layer/fully_connected/kernel/generator_opti/AssignAssign:Decoder/second_layer/fully_connected/kernel/generator_optiLDecoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
?Decoder/second_layer/fully_connected/kernel/generator_opti/readIdentity:Decoder/second_layer/fully_connected/kernel/generator_opti*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
^Decoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
TDecoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NDecoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill^Decoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorTDecoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
<Decoder/second_layer/fully_connected/kernel/generator_opti_1
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel
�
CDecoder/second_layer/fully_connected/kernel/generator_opti_1/AssignAssign<Decoder/second_layer/fully_connected/kernel/generator_opti_1NDecoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
ADecoder/second_layer/fully_connected/kernel/generator_opti_1/readIdentity<Decoder/second_layer/fully_connected/kernel/generator_opti_1*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
JDecoder/second_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
8Decoder/second_layer/fully_connected/bias/generator_opti
VariableV2*
shared_name *<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
?Decoder/second_layer/fully_connected/bias/generator_opti/AssignAssign8Decoder/second_layer/fully_connected/bias/generator_optiJDecoder/second_layer/fully_connected/bias/generator_opti/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
=Decoder/second_layer/fully_connected/bias/generator_opti/readIdentity8Decoder/second_layer/fully_connected/bias/generator_opti*
_output_shapes	
:�*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias
�
LDecoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
:Decoder/second_layer/fully_connected/bias/generator_opti_1
VariableV2*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
ADecoder/second_layer/fully_connected/bias/generator_opti_1/AssignAssign:Decoder/second_layer/fully_connected/bias/generator_opti_1LDecoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
?Decoder/second_layer/fully_connected/bias/generator_opti_1/readIdentity:Decoder/second_layer/fully_connected/bias/generator_opti_1*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
_output_shapes	
:�
�
ODecoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zerosConst*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
=Decoder/second_layer/batch_normalization/gamma/generator_opti
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
	container *
shape:�
�
DDecoder/second_layer/batch_normalization/gamma/generator_opti/AssignAssign=Decoder/second_layer/batch_normalization/gamma/generator_optiODecoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
validate_shape(
�
BDecoder/second_layer/batch_normalization/gamma/generator_opti/readIdentity=Decoder/second_layer/batch_normalization/gamma/generator_opti*
_output_shapes	
:�*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma
�
QDecoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zerosConst*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
?Decoder/second_layer/batch_normalization/gamma/generator_opti_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
	container *
shape:�
�
FDecoder/second_layer/batch_normalization/gamma/generator_opti_1/AssignAssign?Decoder/second_layer/batch_normalization/gamma/generator_opti_1QDecoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zeros*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
DDecoder/second_layer/batch_normalization/gamma/generator_opti_1/readIdentity?Decoder/second_layer/batch_normalization/gamma/generator_opti_1*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
NDecoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zerosConst*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
<Decoder/second_layer/batch_normalization/beta/generator_opti
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
	container *
shape:�
�
CDecoder/second_layer/batch_normalization/beta/generator_opti/AssignAssign<Decoder/second_layer/batch_normalization/beta/generator_optiNDecoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
ADecoder/second_layer/batch_normalization/beta/generator_opti/readIdentity<Decoder/second_layer/batch_normalization/beta/generator_opti*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
PDecoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zerosConst*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
>Decoder/second_layer/batch_normalization/beta/generator_opti_1
VariableV2*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
EDecoder/second_layer/batch_normalization/beta/generator_opti_1/AssignAssign>Decoder/second_layer/batch_normalization/beta/generator_opti_1PDecoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
CDecoder/second_layer/batch_normalization/beta/generator_opti_1/readIdentity>Decoder/second_layer/batch_normalization/beta/generator_opti_1*
_output_shapes	
:�*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta
�
EDecoder/dense/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@Decoder/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
;Decoder/dense/kernel/generator_opti/Initializer/zeros/ConstConst*'
_class
loc:@Decoder/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
5Decoder/dense/kernel/generator_opti/Initializer/zerosFillEDecoder/dense/kernel/generator_opti/Initializer/zeros/shape_as_tensor;Decoder/dense/kernel/generator_opti/Initializer/zeros/Const*'
_class
loc:@Decoder/dense/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
#Decoder/dense/kernel/generator_opti
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *'
_class
loc:@Decoder/dense/kernel*
	container 
�
*Decoder/dense/kernel/generator_opti/AssignAssign#Decoder/dense/kernel/generator_opti5Decoder/dense/kernel/generator_opti/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*'
_class
loc:@Decoder/dense/kernel*
validate_shape(
�
(Decoder/dense/kernel/generator_opti/readIdentity#Decoder/dense/kernel/generator_opti*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:
��
�
GDecoder/dense/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@Decoder/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
=Decoder/dense/kernel/generator_opti_1/Initializer/zeros/ConstConst*'
_class
loc:@Decoder/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7Decoder/dense/kernel/generator_opti_1/Initializer/zerosFillGDecoder/dense/kernel/generator_opti_1/Initializer/zeros/shape_as_tensor=Decoder/dense/kernel/generator_opti_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*'
_class
loc:@Decoder/dense/kernel*

index_type0
�
%Decoder/dense/kernel/generator_opti_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *'
_class
loc:@Decoder/dense/kernel*
	container *
shape:
��
�
,Decoder/dense/kernel/generator_opti_1/AssignAssign%Decoder/dense/kernel/generator_opti_17Decoder/dense/kernel/generator_opti_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Decoder/dense/kernel*
validate_shape(* 
_output_shapes
:
��
�
*Decoder/dense/kernel/generator_opti_1/readIdentity%Decoder/dense/kernel/generator_opti_1*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:
��
�
3Decoder/dense/bias/generator_opti/Initializer/zerosConst*%
_class
loc:@Decoder/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!Decoder/dense/bias/generator_opti
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *%
_class
loc:@Decoder/dense/bias*
	container *
shape:�
�
(Decoder/dense/bias/generator_opti/AssignAssign!Decoder/dense/bias/generator_opti3Decoder/dense/bias/generator_opti/Initializer/zeros*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
&Decoder/dense/bias/generator_opti/readIdentity!Decoder/dense/bias/generator_opti*
_output_shapes	
:�*
T0*%
_class
loc:@Decoder/dense/bias
�
5Decoder/dense/bias/generator_opti_1/Initializer/zerosConst*%
_class
loc:@Decoder/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
#Decoder/dense/bias/generator_opti_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *%
_class
loc:@Decoder/dense/bias
�
*Decoder/dense/bias/generator_opti_1/AssignAssign#Decoder/dense/bias/generator_opti_15Decoder/dense/bias/generator_opti_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes	
:�
�
(Decoder/dense/bias/generator_opti_1/readIdentity#Decoder/dense/bias/generator_opti_1*
T0*%
_class
loc:@Decoder/dense/bias*
_output_shapes	
:�
a
generator_opti/learning_rateConst*
valueB
 *�Q9*
dtype0*
_output_shapes
: 
Y
generator_opti/beta1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Y
generator_opti/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
[
generator_opti/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Jgenerator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam*Encoder/first_layer/fully_connected/kernel9Encoder/first_layer/fully_connected/kernel/generator_opti;Encoder/first_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonVgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
use_locking( *
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
use_nesterov( 
�
Hgenerator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam(Encoder/first_layer/fully_connected/bias7Encoder/first_layer/fully_connected/bias/generator_opti9Encoder/first_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
Kgenerator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Encoder/second_layer/fully_connected/kernel:Encoder/second_layer/fully_connected/kernel/generator_opti<Encoder/second_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0
�
Igenerator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Encoder/second_layer/fully_connected/bias8Encoder/second_layer/fully_connected/bias/generator_opti:Encoder/second_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonXgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias
�
Ngenerator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Encoder/second_layer/batch_normalization/gamma=Encoder/second_layer/batch_normalization/gamma/generator_opti?Encoder/second_layer/batch_normalization/gamma/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�
�
Mgenerator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Encoder/second_layer/batch_normalization/beta<Encoder/second_layer/batch_normalization/beta/generator_opti>Encoder/second_layer/batch_normalization/beta/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilon`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:�
�
9generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdam	ApplyAdamEncoder/encoder_mu/kernel(Encoder/encoder_mu/kernel/generator_opti*Encoder/encoder_mu/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonEgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
use_nesterov( *
_output_shapes
:	�d
�
7generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam	ApplyAdamEncoder/encoder_mu/bias&Encoder/encoder_mu/bias/generator_opti(Encoder/encoder_mu/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonFgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency_1**
_class 
loc:@Encoder/encoder_mu/bias*
use_nesterov( *
_output_shapes
:d*
use_locking( *
T0
�
=generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam	ApplyAdamEncoder/encoder_logvar/kernel,Encoder/encoder_logvar/kernel/generator_opti.Encoder/encoder_logvar/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonIgradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
use_nesterov( *
_output_shapes
:	�d
�
;generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam	ApplyAdamEncoder/encoder_logvar/bias*Encoder/encoder_logvar/bias/generator_opti,Encoder/encoder_logvar/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonJgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
use_nesterov( *
_output_shapes
:d
�
Jgenerator_opti/update_Decoder/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam*Decoder/first_layer/fully_connected/kernel9Decoder/first_layer/fully_connected/kernel/generator_opti;Decoder/first_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonVgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	d�
�
Hgenerator_opti/update_Decoder/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam(Decoder/first_layer/fully_connected/bias7Decoder/first_layer/fully_connected/bias/generator_opti9Decoder/first_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
Kgenerator_opti/update_Decoder/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Decoder/second_layer/fully_connected/kernel:Decoder/second_layer/fully_connected/kernel/generator_opti<Decoder/second_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0
�
Igenerator_opti/update_Decoder/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Decoder/second_layer/fully_connected/bias8Decoder/second_layer/fully_connected/bias/generator_opti:Decoder/second_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonXgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
Ngenerator_opti/update_Decoder/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Decoder/second_layer/batch_normalization/gamma=Decoder/second_layer/batch_normalization/gamma/generator_opti?Decoder/second_layer/batch_normalization/gamma/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�
�
Mgenerator_opti/update_Decoder/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Decoder/second_layer/batch_normalization/beta<Decoder/second_layer/batch_normalization/beta/generator_opti>Decoder/second_layer/batch_normalization/beta/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilon`gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes	
:�*
use_locking( *
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
use_nesterov( 
�
4generator_opti/update_Decoder/dense/kernel/ApplyAdam	ApplyAdamDecoder/dense/kernel#Decoder/dense/kernel/generator_opti%Decoder/dense/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilon@gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
use_locking( *
T0*'
_class
loc:@Decoder/dense/kernel*
use_nesterov( 
�
2generator_opti/update_Decoder/dense/bias/ApplyAdam	ApplyAdamDecoder/dense/bias!Decoder/dense/bias/generator_opti#Decoder/dense/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonAgradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@Decoder/dense/bias*
use_nesterov( *
_output_shapes	
:�
�
generator_opti/mulMulbeta1_power_1/readgenerator_opti/beta13^generator_opti/update_Decoder/dense/bias/ApplyAdam5^generator_opti/update_Decoder/dense/kernel/ApplyAdamI^generator_opti/update_Decoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Decoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Decoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Decoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Decoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Decoder/second_layer/fully_connected/kernel/ApplyAdam<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*%
_class
loc:@Decoder/dense/bias
�
generator_opti/AssignAssignbeta1_power_1generator_opti/mul*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
generator_opti/mul_1Mulbeta2_power_1/readgenerator_opti/beta23^generator_opti/update_Decoder/dense/bias/ApplyAdam5^generator_opti/update_Decoder/dense/kernel/ApplyAdamI^generator_opti/update_Decoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Decoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Decoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Decoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Decoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Decoder/second_layer/fully_connected/kernel/ApplyAdam<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam*
T0*%
_class
loc:@Decoder/dense/bias*
_output_shapes
: 
�
generator_opti/Assign_1Assignbeta2_power_1generator_opti/mul_1*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�

generator_optiNoOp^generator_opti/Assign^generator_opti/Assign_13^generator_opti/update_Decoder/dense/bias/ApplyAdam5^generator_opti/update_Decoder/dense/kernel/ApplyAdamI^generator_opti/update_Decoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Decoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Decoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Decoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Decoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Decoder/second_layer/fully_connected/kernel/ApplyAdam<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam
i
Merge/MergeSummaryMergeSummarygenerator_loss_1discriminator_loss*
N*
_output_shapes
: "�Mv+�     ��	՞����AJ��
��
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
.
Log1p
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
.
Rsqrt
x"T
y"T"
Ttype:

2
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
/
Sign
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.12.02v1.12.0-0-ga6d8ffae09��
�
Encoder/real_inPlaceholder*
dtype0*/
_output_shapes
:���������*$
shape:���������
f
Encoder/Reshape/shapeConst*
valueB"����  *
dtype0*
_output_shapes
:
�
Encoder/ReshapeReshapeEncoder/real_inEncoder/Reshape/shape*
Tshape0*(
_output_shapes
:����������*
T0
�
KEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *HY��*
dtype0*
_output_shapes
: 
�
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *HY�=*
dtype0*
_output_shapes
: 
�
SEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformKEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
seed2 
�
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
_output_shapes
: 
�
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulSEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
EEncoder/first_layer/fully_connected/kernel/Initializer/random_uniformAddIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
*Encoder/first_layer/fully_connected/kernel
VariableV2*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
1Encoder/first_layer/fully_connected/kernel/AssignAssign*Encoder/first_layer/fully_connected/kernelEEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
/Encoder/first_layer/fully_connected/kernel/readIdentity*Encoder/first_layer/fully_connected/kernel*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
:Encoder/first_layer/fully_connected/bias/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
(Encoder/first_layer/fully_connected/bias
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias
�
/Encoder/first_layer/fully_connected/bias/AssignAssign(Encoder/first_layer/fully_connected/bias:Encoder/first_layer/fully_connected/bias/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(
�
-Encoder/first_layer/fully_connected/bias/readIdentity(Encoder/first_layer/fully_connected/bias*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:�*
T0
�
*Encoder/first_layer/fully_connected/MatMulMatMulEncoder/Reshape/Encoder/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
+Encoder/first_layer/fully_connected/BiasAddBiasAdd*Encoder/first_layer/fully_connected/MatMul-Encoder/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
i
$Encoder/first_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
"Encoder/first_layer/leaky_relu/mulMul$Encoder/first_layer/leaky_relu/alpha+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Encoder/first_layer/leaky_reluMaximum"Encoder/first_layer/leaky_relu/mul+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
LEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *qĜ=*
dtype0
�
TEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
��*

seed *
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
�
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
�
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
�
FEncoder/second_layer/fully_connected/kernel/Initializer/random_uniformAddJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
+Encoder/second_layer/fully_connected/kernel
VariableV2* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0
�
2Encoder/second_layer/fully_connected/kernel/AssignAssign+Encoder/second_layer/fully_connected/kernelFEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
0Encoder/second_layer/fully_connected/kernel/readIdentity+Encoder/second_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
;Encoder/second_layer/fully_connected/bias/Initializer/zerosConst*
_output_shapes	
:�*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB�*    *
dtype0
�
)Encoder/second_layer/fully_connected/bias
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container 
�
0Encoder/second_layer/fully_connected/bias/AssignAssign)Encoder/second_layer/fully_connected/bias;Encoder/second_layer/fully_connected/bias/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(
�
.Encoder/second_layer/fully_connected/bias/readIdentity)Encoder/second_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
_output_shapes	
:�
�
+Encoder/second_layer/fully_connected/MatMulMatMulEncoder/first_layer/leaky_relu0Encoder/second_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
,Encoder/second_layer/fully_connected/BiasAddBiasAdd+Encoder/second_layer/fully_connected/MatMul.Encoder/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
?Encoder/second_layer/batch_normalization/gamma/Initializer/onesConst*
_output_shapes	
:�*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB�*  �?*
dtype0
�
.Encoder/second_layer/batch_normalization/gamma
VariableV2*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0
�
5Encoder/second_layer/batch_normalization/gamma/AssignAssign.Encoder/second_layer/batch_normalization/gamma?Encoder/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
3Encoder/second_layer/batch_normalization/gamma/readIdentity.Encoder/second_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
?Encoder/second_layer/batch_normalization/beta/Initializer/zerosConst*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
-Encoder/second_layer/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
	container *
shape:�
�
4Encoder/second_layer/batch_normalization/beta/AssignAssign-Encoder/second_layer/batch_normalization/beta?Encoder/second_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
2Encoder/second_layer/batch_normalization/beta/readIdentity-Encoder/second_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
FEncoder/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4Encoder/second_layer/batch_normalization/moving_mean
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean
�
;Encoder/second_layer/batch_normalization/moving_mean/AssignAssign4Encoder/second_layer/batch_normalization/moving_meanFEncoder/second_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:�
�
9Encoder/second_layer/batch_normalization/moving_mean/readIdentity4Encoder/second_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
_output_shapes	
:�
�
IEncoder/second_layer/batch_normalization/moving_variance/Initializer/onesConst*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
8Encoder/second_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
	container *
shape:�
�
?Encoder/second_layer/batch_normalization/moving_variance/AssignAssign8Encoder/second_layer/batch_normalization/moving_varianceIEncoder/second_layer/batch_normalization/moving_variance/Initializer/ones*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
=Encoder/second_layer/batch_normalization/moving_variance/readIdentity8Encoder/second_layer/batch_normalization/moving_variance*
T0*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
_output_shapes	
:�
}
8Encoder/second_layer/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
6Encoder/second_layer/batch_normalization/batchnorm/addAdd=Encoder/second_layer/batch_normalization/moving_variance/read8Encoder/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:�
�
8Encoder/second_layer/batch_normalization/batchnorm/RsqrtRsqrt6Encoder/second_layer/batch_normalization/batchnorm/add*
_output_shapes	
:�*
T0
�
6Encoder/second_layer/batch_normalization/batchnorm/mulMul8Encoder/second_layer/batch_normalization/batchnorm/Rsqrt3Encoder/second_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
�
8Encoder/second_layer/batch_normalization/batchnorm/mul_1Mul,Encoder/second_layer/fully_connected/BiasAdd6Encoder/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
�
8Encoder/second_layer/batch_normalization/batchnorm/mul_2Mul9Encoder/second_layer/batch_normalization/moving_mean/read6Encoder/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:�*
T0
�
6Encoder/second_layer/batch_normalization/batchnorm/subSub2Encoder/second_layer/batch_normalization/beta/read8Encoder/second_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
8Encoder/second_layer/batch_normalization/batchnorm/add_1Add8Encoder/second_layer/batch_normalization/batchnorm/mul_16Encoder/second_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:����������
j
%Encoder/second_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
#Encoder/second_layer/leaky_relu/mulMul%Encoder/second_layer/leaky_relu/alpha8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Encoder/second_layer/leaky_reluMaximum#Encoder/second_layer/leaky_relu/mul8Encoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
:Encoder/encoder_mu/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
�
8Encoder/encoder_mu/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
�
8Encoder/encoder_mu/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *?��=*
dtype0*
_output_shapes
: 
�
BEncoder/encoder_mu/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Encoder/encoder_mu/kernel/Initializer/random_uniform/shape*
_output_shapes
:	�d*

seed *
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
seed2 *
dtype0
�
8Encoder/encoder_mu/kernel/Initializer/random_uniform/subSub8Encoder/encoder_mu/kernel/Initializer/random_uniform/max8Encoder/encoder_mu/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
: 
�
8Encoder/encoder_mu/kernel/Initializer/random_uniform/mulMulBEncoder/encoder_mu/kernel/Initializer/random_uniform/RandomUniform8Encoder/encoder_mu/kernel/Initializer/random_uniform/sub*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	�d*
T0
�
4Encoder/encoder_mu/kernel/Initializer/random_uniformAdd8Encoder/encoder_mu/kernel/Initializer/random_uniform/mul8Encoder/encoder_mu/kernel/Initializer/random_uniform/min*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	�d*
T0
�
Encoder/encoder_mu/kernel
VariableV2*
shared_name *,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d
�
 Encoder/encoder_mu/kernel/AssignAssignEncoder/encoder_mu/kernel4Encoder/encoder_mu/kernel/Initializer/random_uniform*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0
�
Encoder/encoder_mu/kernel/readIdentityEncoder/encoder_mu/kernel*
_output_shapes
:	�d*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel
�
)Encoder/encoder_mu/bias/Initializer/zerosConst**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0*
_output_shapes
:d
�
Encoder/encoder_mu/bias
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name **
_class 
loc:@Encoder/encoder_mu/bias*
	container 
�
Encoder/encoder_mu/bias/AssignAssignEncoder/encoder_mu/bias)Encoder/encoder_mu/bias/Initializer/zeros**
_class 
loc:@Encoder/encoder_mu/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
�
Encoder/encoder_mu/bias/readIdentityEncoder/encoder_mu/bias*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d
�
Encoder/encoder_mu/MatMulMatMulEncoder/second_layer/leaky_reluEncoder/encoder_mu/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
�
Encoder/encoder_mu/BiasAddBiasAddEncoder/encoder_mu/MatMulEncoder/encoder_mu/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������d
�
>Encoder/encoder_logvar/kernel/Initializer/random_uniform/shapeConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
�
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/minConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
�
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/maxConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *?��=*
dtype0*
_output_shapes
: 
�
FEncoder/encoder_logvar/kernel/Initializer/random_uniform/RandomUniformRandomUniform>Encoder/encoder_logvar/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	�d*

seed *
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
seed2 
�
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/subSub<Encoder/encoder_logvar/kernel/Initializer/random_uniform/max<Encoder/encoder_logvar/kernel/Initializer/random_uniform/min*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
: *
T0
�
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/mulMulFEncoder/encoder_logvar/kernel/Initializer/random_uniform/RandomUniform<Encoder/encoder_logvar/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	�d
�
8Encoder/encoder_logvar/kernel/Initializer/random_uniformAdd<Encoder/encoder_logvar/kernel/Initializer/random_uniform/mul<Encoder/encoder_logvar/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	�d
�
Encoder/encoder_logvar/kernel
VariableV2*
dtype0*
_output_shapes
:	�d*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	�d
�
$Encoder/encoder_logvar/kernel/AssignAssignEncoder/encoder_logvar/kernel8Encoder/encoder_logvar/kernel/Initializer/random_uniform*
use_locking(*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(*
_output_shapes
:	�d
�
"Encoder/encoder_logvar/kernel/readIdentityEncoder/encoder_logvar/kernel*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	�d
�
-Encoder/encoder_logvar/bias/Initializer/zerosConst*
_output_shapes
:d*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0
�
Encoder/encoder_logvar/bias
VariableV2*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name *.
_class$
" loc:@Encoder/encoder_logvar/bias
�
"Encoder/encoder_logvar/bias/AssignAssignEncoder/encoder_logvar/bias-Encoder/encoder_logvar/bias/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
:d
�
 Encoder/encoder_logvar/bias/readIdentityEncoder/encoder_logvar/bias*
_output_shapes
:d*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias
�
Encoder/encoder_logvar/MatMulMatMulEncoder/second_layer/leaky_relu"Encoder/encoder_logvar/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
�
Encoder/encoder_logvar/BiasAddBiasAddEncoder/encoder_logvar/MatMul Encoder/encoder_logvar/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������d
e
Encoder/random_normal/shapeConst*
_output_shapes
:*
valueB	Rd*
dtype0	
_
Encoder/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
Encoder/random_normal/stddevConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
*Encoder/random_normal/RandomStandardNormalRandomStandardNormalEncoder/random_normal/shape*
dtype0*
_output_shapes
:d*
seed2 *

seed *
T0	
�
Encoder/random_normal/mulMul*Encoder/random_normal/RandomStandardNormalEncoder/random_normal/stddev*
_output_shapes
:d*
T0
x
Encoder/random_normalAddEncoder/random_normal/mulEncoder/random_normal/mean*
T0*
_output_shapes
:d
V
Encoder/truediv/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0

Encoder/truedivRealDivEncoder/encoder_logvar/BiasAddEncoder/truediv/y*
T0*'
_output_shapes
:���������d
U
Encoder/ExpExpEncoder/truediv*'
_output_shapes
:���������d*
T0
o
Encoder/logvar_stdMulEncoder/random_normalEncoder/Exp*
T0*'
_output_shapes
:���������d
t
Encoder/AddAddEncoder/logvar_stdEncoder/encoder_mu/BiasAdd*
T0*'
_output_shapes
:���������d
^
Encoder/encoder_codeSigmoidEncoder/Add*
T0*'
_output_shapes
:���������d
�
KDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB"d      *
dtype0
�
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
�
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *?��=*
dtype0
�
SDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformKDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*

seed *
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
seed2 *
dtype0*
_output_shapes
:	d�
�
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
: 
�
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulSDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d�*
T0
�
EDecoder/first_layer/fully_connected/kernel/Initializer/random_uniformAddIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
*Decoder/first_layer/fully_connected/kernel
VariableV2*
dtype0*
_output_shapes
:	d�*
shared_name *=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
	container *
shape:	d�
�
1Decoder/first_layer/fully_connected/kernel/AssignAssign*Decoder/first_layer/fully_connected/kernelEDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
/Decoder/first_layer/fully_connected/kernel/readIdentity*Decoder/first_layer/fully_connected/kernel*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d�*
T0
�
:Decoder/first_layer/fully_connected/bias/Initializer/zerosConst*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
(Decoder/first_layer/fully_connected/bias
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
	container 
�
/Decoder/first_layer/fully_connected/bias/AssignAssign(Decoder/first_layer/fully_connected/bias:Decoder/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
-Decoder/first_layer/fully_connected/bias/readIdentity(Decoder/first_layer/fully_connected/bias*
_output_shapes	
:�*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias
�
*Decoder/first_layer/fully_connected/MatMulMatMulEncoder/encoder_code/Decoder/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
+Decoder/first_layer/fully_connected/BiasAddBiasAdd*Decoder/first_layer/fully_connected/MatMul-Decoder/first_layer/fully_connected/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
i
$Decoder/first_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
"Decoder/first_layer/leaky_relu/mulMul$Decoder/first_layer/leaky_relu/alpha+Decoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Decoder/first_layer/leaky_reluMaximum"Decoder/first_layer/leaky_relu/mul+Decoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
LDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0
�
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *qĜ=*
dtype0*
_output_shapes
: 
�
TDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*

seed *
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
��
�
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
_output_shapes
: *
T0
�
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
FDecoder/second_layer/fully_connected/kernel/Initializer/random_uniformAddJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
+Decoder/second_layer/fully_connected/kernel
VariableV2*
shared_name *>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
2Decoder/second_layer/fully_connected/kernel/AssignAssign+Decoder/second_layer/fully_connected/kernelFDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
0Decoder/second_layer/fully_connected/kernel/readIdentity+Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel
�
;Decoder/second_layer/fully_connected/bias/Initializer/zerosConst*
_output_shapes	
:�*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
valueB�*    *
dtype0
�
)Decoder/second_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
	container *
shape:�
�
0Decoder/second_layer/fully_connected/bias/AssignAssign)Decoder/second_layer/fully_connected/bias;Decoder/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
.Decoder/second_layer/fully_connected/bias/readIdentity)Decoder/second_layer/fully_connected/bias*
_output_shapes	
:�*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias
�
+Decoder/second_layer/fully_connected/MatMulMatMulDecoder/first_layer/leaky_relu0Decoder/second_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
,Decoder/second_layer/fully_connected/BiasAddBiasAdd+Decoder/second_layer/fully_connected/MatMul.Decoder/second_layer/fully_connected/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
�
?Decoder/second_layer/batch_normalization/gamma/Initializer/onesConst*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
.Decoder/second_layer/batch_normalization/gamma
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma
�
5Decoder/second_layer/batch_normalization/gamma/AssignAssign.Decoder/second_layer/batch_normalization/gamma?Decoder/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
3Decoder/second_layer/batch_normalization/gamma/readIdentity.Decoder/second_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
?Decoder/second_layer/batch_normalization/beta/Initializer/zerosConst*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
-Decoder/second_layer/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
	container *
shape:�
�
4Decoder/second_layer/batch_normalization/beta/AssignAssign-Decoder/second_layer/batch_normalization/beta?Decoder/second_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
2Decoder/second_layer/batch_normalization/beta/readIdentity-Decoder/second_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
FDecoder/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*
_output_shapes	
:�*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
valueB�*    *
dtype0
�
4Decoder/second_layer/batch_normalization/moving_mean
VariableV2*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
;Decoder/second_layer/batch_normalization/moving_mean/AssignAssign4Decoder/second_layer/batch_normalization/moving_meanFDecoder/second_layer/batch_normalization/moving_mean/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
validate_shape(
�
9Decoder/second_layer/batch_normalization/moving_mean/readIdentity4Decoder/second_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
_output_shapes	
:�
�
IDecoder/second_layer/batch_normalization/moving_variance/Initializer/onesConst*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
8Decoder/second_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
	container *
shape:�
�
?Decoder/second_layer/batch_normalization/moving_variance/AssignAssign8Decoder/second_layer/batch_normalization/moving_varianceIDecoder/second_layer/batch_normalization/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance
�
=Decoder/second_layer/batch_normalization/moving_variance/readIdentity8Decoder/second_layer/batch_normalization/moving_variance*
T0*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
_output_shapes	
:�
}
8Decoder/second_layer/batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
6Decoder/second_layer/batch_normalization/batchnorm/addAdd=Decoder/second_layer/batch_normalization/moving_variance/read8Decoder/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:�
�
8Decoder/second_layer/batch_normalization/batchnorm/RsqrtRsqrt6Decoder/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:�
�
6Decoder/second_layer/batch_normalization/batchnorm/mulMul8Decoder/second_layer/batch_normalization/batchnorm/Rsqrt3Decoder/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
8Decoder/second_layer/batch_normalization/batchnorm/mul_1Mul,Decoder/second_layer/fully_connected/BiasAdd6Decoder/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
8Decoder/second_layer/batch_normalization/batchnorm/mul_2Mul9Decoder/second_layer/batch_normalization/moving_mean/read6Decoder/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:�*
T0
�
6Decoder/second_layer/batch_normalization/batchnorm/subSub2Decoder/second_layer/batch_normalization/beta/read8Decoder/second_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
8Decoder/second_layer/batch_normalization/batchnorm/add_1Add8Decoder/second_layer/batch_normalization/batchnorm/mul_16Decoder/second_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:����������
j
%Decoder/second_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
#Decoder/second_layer/leaky_relu/mulMul%Decoder/second_layer/leaky_relu/alpha8Decoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Decoder/second_layer/leaky_reluMaximum#Decoder/second_layer/leaky_relu/mul8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
5Decoder/dense/kernel/Initializer/random_uniform/shapeConst*'
_class
loc:@Decoder/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
3Decoder/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *'
_class
loc:@Decoder/dense/kernel*
valueB
 *HY��
�
3Decoder/dense/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@Decoder/dense/kernel*
valueB
 *HY�=*
dtype0*
_output_shapes
: 
�
=Decoder/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5Decoder/dense/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@Decoder/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
��*

seed 
�
3Decoder/dense/kernel/Initializer/random_uniform/subSub3Decoder/dense/kernel/Initializer/random_uniform/max3Decoder/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@Decoder/dense/kernel*
_output_shapes
: 
�
3Decoder/dense/kernel/Initializer/random_uniform/mulMul=Decoder/dense/kernel/Initializer/random_uniform/RandomUniform3Decoder/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:
��
�
/Decoder/dense/kernel/Initializer/random_uniformAdd3Decoder/dense/kernel/Initializer/random_uniform/mul3Decoder/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:
��
�
Decoder/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *'
_class
loc:@Decoder/dense/kernel*
	container *
shape:
��
�
Decoder/dense/kernel/AssignAssignDecoder/dense/kernel/Decoder/dense/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*'
_class
loc:@Decoder/dense/kernel
�
Decoder/dense/kernel/readIdentityDecoder/dense/kernel*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:
��
�
$Decoder/dense/bias/Initializer/zerosConst*%
_class
loc:@Decoder/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Decoder/dense/bias
VariableV2*
shared_name *%
_class
loc:@Decoder/dense/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
Decoder/dense/bias/AssignAssignDecoder/dense/bias$Decoder/dense/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes	
:�
�
Decoder/dense/bias/readIdentityDecoder/dense/bias*
T0*%
_class
loc:@Decoder/dense/bias*
_output_shapes	
:�
�
Decoder/dense/MatMulMatMulDecoder/second_layer/leaky_reluDecoder/dense/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
Decoder/dense/BiasAddBiasAddDecoder/dense/MatMulDecoder/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
d
Decoder/last_layerTanhDecoder/dense/BiasAdd*
T0*(
_output_shapes
:����������
t
Decoder/reshape_image/shapeConst*%
valueB"����         *
dtype0*
_output_shapes
:
�
Decoder/reshape_imageReshapeDecoder/last_layerDecoder/reshape_image/shape*
T0*
Tshape0*/
_output_shapes
:���������
~
Discriminator/noise_code_inPlaceholder*
dtype0*'
_output_shapes
:���������d*
shape:���������d
�
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *?��=*
dtype0*
_output_shapes
: 
�
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d�*

seed *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
seed2 
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
: 
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
_output_shapes
:	d�*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
0Discriminator/first_layer/fully_connected/kernel
VariableV2*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�
�
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
@Discriminator/first_layer/fully_connected/bias/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
.Discriminator/first_layer/fully_connected/bias
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
5Discriminator/first_layer/fully_connected/bias/AssignAssign.Discriminator/first_layer/fully_connected/bias@Discriminator/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:�
�
0Discriminator/first_layer/fully_connected/MatMulMatMulEncoder/encoder_code5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
o
*Discriminator/first_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
$Discriminator/first_layer/leaky_reluMaximum(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
RDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *��=*
dtype0*
_output_shapes
: 
�
ZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformRDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
��*

seed 
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
1Discriminator/second_layer/fully_connected/kernel
VariableV2*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
8Discriminator/second_layer/fully_connected/kernel/AssignAssign1Discriminator/second_layer/fully_connected/kernelLDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
6Discriminator/second_layer/fully_connected/kernel/readIdentity1Discriminator/second_layer/fully_connected/kernel*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
ADiscriminator/second_layer/fully_connected/bias/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
/Discriminator/second_layer/fully_connected/bias
VariableV2*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
6Discriminator/second_layer/fully_connected/bias/AssignAssign/Discriminator/second_layer/fully_connected/biasADiscriminator/second_layer/fully_connected/bias/Initializer/zeros*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:�*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
2Discriminator/second_layer/fully_connected/BiasAddBiasAdd1Discriminator/second_layer/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
p
+Discriminator/second_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
)Discriminator/second_layer/leaky_relu/mulMul+Discriminator/second_layer/leaky_relu/alpha2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
%Discriminator/second_layer/leaky_reluMaximum)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
:Discriminator/prob/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
8Discriminator/prob/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Iv�*
dtype0*
_output_shapes
: 
�
8Discriminator/prob/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Iv>
�
BDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Discriminator/prob/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	�*

seed *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
seed2 
�
8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
: 
�
8Discriminator/prob/kernel/Initializer/random_uniform/mulMulBDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniform8Discriminator/prob/kernel/Initializer/random_uniform/sub*
_output_shapes
:	�*
T0*,
_class"
 loc:@Discriminator/prob/kernel
�
4Discriminator/prob/kernel/Initializer/random_uniformAdd8Discriminator/prob/kernel/Initializer/random_uniform/mul8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
Discriminator/prob/kernel
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	�
�
 Discriminator/prob/kernel/AssignAssignDiscriminator/prob/kernel4Discriminator/prob/kernel/Initializer/random_uniform*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
Discriminator/prob/kernel/readIdentityDiscriminator/prob/kernel*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
)Discriminator/prob/bias/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
�
Discriminator/prob/bias
VariableV2*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
Discriminator/prob/bias/AssignAssignDiscriminator/prob/bias)Discriminator/prob/bias/Initializer/zeros*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
Discriminator/prob/bias/readIdentityDiscriminator/prob/bias*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
�
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
�
2Discriminator/first_layer_1/fully_connected/MatMulMatMulDiscriminator/noise_code_in5Discriminator/first_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
3Discriminator/first_layer_1/fully_connected/BiasAddBiasAdd2Discriminator/first_layer_1/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
q
,Discriminator/first_layer_1/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
*Discriminator/first_layer_1/leaky_relu/mulMul,Discriminator/first_layer_1/leaky_relu/alpha3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
&Discriminator/first_layer_1/leaky_reluMaximum*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
3Discriminator/second_layer_1/fully_connected/MatMulMatMul&Discriminator/first_layer_1/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
4Discriminator/second_layer_1/fully_connected/BiasAddBiasAdd3Discriminator/second_layer_1/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
r
-Discriminator/second_layer_1/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
�
+Discriminator/second_layer_1/leaky_relu/mulMul-Discriminator/second_layer_1/leaky_relu/alpha4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
'Discriminator/second_layer_1/leaky_reluMaximum+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Discriminator/prob_1/MatMulMatMul'Discriminator/second_layer_1/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Discriminator/prob_1/BiasAddBiasAddDiscriminator/prob_1/MatMulDiscriminator/prob/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
i
ones_like/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
T
ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
w
	ones_likeFillones_like/Shapeones_like/Const*
T0*

index_type0*'
_output_shapes
:���������
s
logistic_loss/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:���������
�
logistic_loss/SelectSelectlogistic_loss/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:���������
f
logistic_loss/NegNegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/NegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
q
logistic_loss/mulMulDiscriminator/prob/BiasAdd	ones_like*
T0*'
_output_shapes
:���������
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*'
_output_shapes
:���������*
T0
b
logistic_loss/ExpExplogistic_loss/Select_1*
T0*'
_output_shapes
:���������
a
logistic_loss/Log1pLog1plogistic_loss/Exp*'
_output_shapes
:���������*
T0
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0*'
_output_shapes
:���������
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
m
adversalrial_lossMeanlogistic_lossConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
l
SubSubDecoder/reshape_imageEncoder/real_in*
T0*/
_output_shapes
:���������
I
AbsAbsSub*/
_output_shapes
:���������*
T0
`
Const_1Const*
dtype0*
_output_shapes
:*%
valueB"             
b
pixelwise_lossMeanAbsConst_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
J
mul/xConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
E
mulMulmul/xadversalrial_loss*
T0*
_output_shapes
: 
L
mul_1/xConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
F
mul_1Mulmul_1/xpixelwise_loss*
T0*
_output_shapes
: 
B
generator_lossAddmulmul_1*
T0*
_output_shapes
: 
e
generator_loss_1/tagConst*!
valueB Bgenerator_loss_1*
dtype0*
_output_shapes
: 
k
generator_loss_1HistogramSummarygenerator_loss_1/taggenerator_loss*
T0*
_output_shapes
: 
m
ones_like_1/ShapeShapeDiscriminator/prob_1/BiasAdd*
_output_shapes
:*
T0*
out_type0
V
ones_like_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
}
ones_like_1Fillones_like_1/Shapeones_like_1/Const*
T0*

index_type0*'
_output_shapes
:���������
w
logistic_loss_1/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss_1/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:���������
�
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*'
_output_shapes
:���������*
T0
j
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
w
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAddones_like_1*
T0*'
_output_shapes
:���������
y
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*'
_output_shapes
:���������*
T0
f
logistic_loss_1/ExpExplogistic_loss_1/Select_1*'
_output_shapes
:���������*
T0
e
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*
T0*'
_output_shapes
:���������
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*
T0*'
_output_shapes
:���������
X
Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
d
MeanMeanlogistic_loss_1Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e

zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
u
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:���������*
T0
�
logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:���������*
T0
�
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:���������
h
logistic_loss_2/NegNegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
t
logistic_loss_2/mulMulDiscriminator/prob/BiasAdd
zeros_like*
T0*'
_output_shapes
:���������
y
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*'
_output_shapes
:���������*
T0
f
logistic_loss_2/ExpExplogistic_loss_2/Select_1*
T0*'
_output_shapes
:���������
e
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*
T0*'
_output_shapes
:���������
t
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*
T0*'
_output_shapes
:���������
X
Const_3Const*
dtype0*
_output_shapes
:*
valueB"       
f
Mean_1Meanlogistic_loss_2Const_3*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
9
addAddMeanMean_1*
T0*
_output_shapes
: 
i
discriminator_loss/tagConst*
dtype0*
_output_shapes
: *#
valueB Bdiscriminator_loss
d
discriminator_lossHistogramSummarydiscriminator_loss/tagadd*
_output_shapes
: *
T0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
<
#gradients/add_grad/tuple/group_depsNoOp^gradients/Fill
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Fill$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/Fill$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
r
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
gradients/Mean_grad/ReshapeReshape+gradients/add_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
h
gradients/Mean_grad/ShapeShapelogistic_loss_1*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
j
gradients/Mean_grad/Shape_1Shapelogistic_loss_1*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:���������
t
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients/Mean_1_grad/ReshapeReshape-gradients/add_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
j
gradients/Mean_1_grad/ShapeShapelogistic_loss_2*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
l
gradients/Mean_1_grad/Shape_1Shapelogistic_loss_2*
T0*
out_type0*
_output_shapes
:
`
gradients/Mean_1_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
e
gradients/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
g
gradients/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
a
gradients/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
�
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*'
_output_shapes
:���������
w
$gradients/logistic_loss_1_grad/ShapeShapelogistic_loss_1/sub*
T0*
out_type0*
_output_shapes
:
{
&gradients/logistic_loss_1_grad/Shape_1Shapelogistic_loss_1/Log1p*
T0*
out_type0*
_output_shapes
:
�
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
/gradients/logistic_loss_1_grad/tuple/group_depsNoOp'^gradients/logistic_loss_1_grad/Reshape)^gradients/logistic_loss_1_grad/Reshape_1
�
7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape*'
_output_shapes
:���������
�
9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1*'
_output_shapes
:���������
w
$gradients/logistic_loss_2_grad/ShapeShapelogistic_loss_2/sub*
T0*
out_type0*
_output_shapes
:
{
&gradients/logistic_loss_2_grad/Shape_1Shapelogistic_loss_2/Log1p*
T0*
out_type0*
_output_shapes
:
�
4gradients/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_2_grad/Shape&gradients/logistic_loss_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"gradients/logistic_loss_2_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&gradients/logistic_loss_2_grad/ReshapeReshape"gradients/logistic_loss_2_grad/Sum$gradients/logistic_loss_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
$gradients/logistic_loss_2_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss_2_grad/Reshape_1Reshape$gradients/logistic_loss_2_grad/Sum_1&gradients/logistic_loss_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/gradients/logistic_loss_2_grad/tuple/group_depsNoOp'^gradients/logistic_loss_2_grad/Reshape)^gradients/logistic_loss_2_grad/Reshape_1
�
7gradients/logistic_loss_2_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_2_grad/Reshape0^gradients/logistic_loss_2_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_2_grad/Reshape*'
_output_shapes
:���������
�
9gradients/logistic_loss_2_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_2_grad/Reshape_10^gradients/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_2_grad/Reshape_1*'
_output_shapes
:���������
~
(gradients/logistic_loss_1/sub_grad/ShapeShapelogistic_loss_1/Select*
T0*
out_type0*
_output_shapes
:
}
*gradients/logistic_loss_1/sub_grad/Shape_1Shapelogistic_loss_1/mul*
T0*
out_type0*
_output_shapes
:
�
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients/logistic_loss_1/sub_grad/Sum_1Sum7gradients/logistic_loss_1_grad/tuple/control_dependency:gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
z
&gradients/logistic_loss_1/sub_grad/NegNeg(gradients/logistic_loss_1/sub_grad/Sum_1*
T0*
_output_shapes
:
�
,gradients/logistic_loss_1/sub_grad/Reshape_1Reshape&gradients/logistic_loss_1/sub_grad/Neg*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
3gradients/logistic_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/sub_grad/Reshape-^gradients/logistic_loss_1/sub_grad/Reshape_1
�
;gradients/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/sub_grad/Reshape4^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/sub_grad/Reshape*'
_output_shapes
:���������
�
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1*'
_output_shapes
:���������
�
*gradients/logistic_loss_1/Log1p_grad/add/xConst:^gradients/logistic_loss_1_grad/tuple/control_dependency_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
(gradients/logistic_loss_1/Log1p_grad/addAdd*gradients/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*
T0*'
_output_shapes
:���������
�
/gradients/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*
T0*'
_output_shapes
:���������
�
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*'
_output_shapes
:���������*
T0
~
(gradients/logistic_loss_2/sub_grad/ShapeShapelogistic_loss_2/Select*
T0*
out_type0*
_output_shapes
:
}
*gradients/logistic_loss_2/sub_grad/Shape_1Shapelogistic_loss_2/mul*
_output_shapes
:*
T0*
out_type0
�
8gradients/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_2/sub_grad/Shape*gradients/logistic_loss_2/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients/logistic_loss_2/sub_grad/SumSum7gradients/logistic_loss_2_grad/tuple/control_dependency8gradients/logistic_loss_2/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*gradients/logistic_loss_2/sub_grad/ReshapeReshape&gradients/logistic_loss_2/sub_grad/Sum(gradients/logistic_loss_2/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients/logistic_loss_2/sub_grad/Sum_1Sum7gradients/logistic_loss_2_grad/tuple/control_dependency:gradients/logistic_loss_2/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
z
&gradients/logistic_loss_2/sub_grad/NegNeg(gradients/logistic_loss_2/sub_grad/Sum_1*
_output_shapes
:*
T0
�
,gradients/logistic_loss_2/sub_grad/Reshape_1Reshape&gradients/logistic_loss_2/sub_grad/Neg*gradients/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
3gradients/logistic_loss_2/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_2/sub_grad/Reshape-^gradients/logistic_loss_2/sub_grad/Reshape_1
�
;gradients/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_2/sub_grad/Reshape4^gradients/logistic_loss_2/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_2/sub_grad/Reshape*'
_output_shapes
:���������
�
=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_2/sub_grad/Reshape_14^gradients/logistic_loss_2/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients/logistic_loss_2/sub_grad/Reshape_1
�
*gradients/logistic_loss_2/Log1p_grad/add/xConst:^gradients/logistic_loss_2_grad/tuple/control_dependency_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
(gradients/logistic_loss_2/Log1p_grad/addAdd*gradients/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*'
_output_shapes
:���������*
T0
�
/gradients/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_2/Log1p_grad/add*'
_output_shapes
:���������*
T0
�
(gradients/logistic_loss_2/Log1p_grad/mulMul9gradients/logistic_loss_2_grad/tuple/control_dependency_1/gradients/logistic_loss_2/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
0gradients/logistic_loss_1/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:���������*
T0
�
,gradients/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual;gradients/logistic_loss_1/sub_grad/tuple/control_dependency0gradients/logistic_loss_1/Select_grad/zeros_like*
T0*'
_output_shapes
:���������
�
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
�
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:���������
�
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_grad/Select_1
�
(gradients/logistic_loss_1/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
u
*gradients/logistic_loss_1/mul_grad/Shape_1Shapeones_like_1*
_output_shapes
:*
T0*
out_type0
�
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1ones_like_1*
T0*'
_output_shapes
:���������
�
&gradients/logistic_loss_1/mul_grad/SumSum&gradients/logistic_loss_1/mul_grad/Mul8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*gradients/logistic_loss_1/mul_grad/ReshapeReshape&gradients/logistic_loss_1/mul_grad/Sum(gradients/logistic_loss_1/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients/logistic_loss_1/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
3gradients/logistic_loss_1/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/mul_grad/Reshape-^gradients/logistic_loss_1/mul_grad/Reshape_1
�
;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/mul_grad/Reshape4^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/mul_grad/Reshape*'
_output_shapes
:���������
�
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1*'
_output_shapes
:���������
�
&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0*'
_output_shapes
:���������
�
0gradients/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
,gradients/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual;gradients/logistic_loss_2/sub_grad/tuple/control_dependency0gradients/logistic_loss_2/Select_grad/zeros_like*
T0*'
_output_shapes
:���������
�
.gradients/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual0gradients/logistic_loss_2/Select_grad/zeros_like;gradients/logistic_loss_2/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
6gradients/logistic_loss_2/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_2/Select_grad/Select/^gradients/logistic_loss_2/Select_grad/Select_1
�
>gradients/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_2/Select_grad/Select7^gradients/logistic_loss_2/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*'
_output_shapes
:���������
�
@gradients/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_2/Select_grad/Select_17^gradients/logistic_loss_2/Select_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@gradients/logistic_loss_2/Select_grad/Select_1
�
(gradients/logistic_loss_2/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
t
*gradients/logistic_loss_2/mul_grad/Shape_1Shape
zeros_like*
T0*
out_type0*
_output_shapes
:
�
8gradients/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_2/mul_grad/Shape*gradients/logistic_loss_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients/logistic_loss_2/mul_grad/MulMul=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1
zeros_like*
T0*'
_output_shapes
:���������
�
&gradients/logistic_loss_2/mul_grad/SumSum&gradients/logistic_loss_2/mul_grad/Mul8gradients/logistic_loss_2/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients/logistic_loss_2/mul_grad/ReshapeReshape&gradients/logistic_loss_2/mul_grad/Sum(gradients/logistic_loss_2/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
(gradients/logistic_loss_2/mul_grad/Sum_1Sum(gradients/logistic_loss_2/mul_grad/Mul_1:gradients/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
,gradients/logistic_loss_2/mul_grad/Reshape_1Reshape(gradients/logistic_loss_2/mul_grad/Sum_1*gradients/logistic_loss_2/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
3gradients/logistic_loss_2/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_2/mul_grad/Reshape-^gradients/logistic_loss_2/mul_grad/Reshape_1
�
;gradients/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_2/mul_grad/Reshape4^gradients/logistic_loss_2/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_2/mul_grad/Reshape*'
_output_shapes
:���������
�
=gradients/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_2/mul_grad/Reshape_14^gradients/logistic_loss_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:���������
�
&gradients/logistic_loss_2/Exp_grad/mulMul(gradients/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*
T0*'
_output_shapes
:���������
�
2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*
T0*'
_output_shapes
:���������
�
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*
T0*'
_output_shapes
:���������
�
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*
T0*'
_output_shapes
:���������
�
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
�
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select
�
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1*'
_output_shapes
:���������
�
2gradients/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*'
_output_shapes
:���������*
T0
�
.gradients/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual&gradients/logistic_loss_2/Exp_grad/mul2gradients/logistic_loss_2/Select_1_grad/zeros_like*
T0*'
_output_shapes
:���������
�
0gradients/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients/logistic_loss_2/Select_1_grad/zeros_like&gradients/logistic_loss_2/Exp_grad/mul*
T0*'
_output_shapes
:���������
�
8gradients/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_2/Select_1_grad/Select1^gradients/logistic_loss_2/Select_1_grad/Select_1
�
@gradients/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_2/Select_1_grad/Select9^gradients/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_2/Select_1_grad/Select*'
_output_shapes
:���������
�
Bgradients/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_2/Select_1_grad/Select_19^gradients/logistic_loss_2/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*C
_class9
75loc:@gradients/logistic_loss_2/Select_1_grad/Select_1
�
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
&gradients/logistic_loss_2/Neg_grad/NegNeg@gradients/logistic_loss_2/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
gradients/AddNAddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N*'
_output_shapes
:���������
�
7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:
�
<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN8^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select
�
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
gradients/AddN_1AddN>gradients/logistic_loss_2/Select_grad/tuple/control_dependency;gradients/logistic_loss_2/mul_grad/tuple/control_dependencyBgradients/logistic_loss_2/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_2/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*
N*'
_output_shapes
:���������
�
5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
T0*
data_formatNHWC*
_output_shapes
:
�
:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_16^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad
�
Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*'
_output_shapes
:���������
�
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
;gradients/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp2^gradients/Discriminator/prob_1/MatMul_grad/MatMul4^gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
�
Cgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/Discriminator/prob_1/MatMul_grad/MatMul<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*D
_class:
86loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul
�
Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
1gradients/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
�
9gradients/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp0^gradients/Discriminator/prob/MatMul_grad/MatMul2^gradients/Discriminator/prob/MatMul_grad/MatMul_1
�
Agradients/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity/gradients/Discriminator/prob/MatMul_grad/MatMul:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/Discriminator/prob/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1
�
gradients/AddN_2AddNFgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
�
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosFill>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Cgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape:gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape
�
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
:gradients/Discriminator/second_layer/leaky_relu_grad/zerosFill<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Jgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Discriminator/second_layer/leaky_relu_grad/Shape<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
;gradients/Discriminator/second_layer/leaky_relu_grad/SelectSelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency:gradients/Discriminator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
8gradients/Discriminator/second_layer/leaky_relu_grad/SumSum;gradients/Discriminator/second_layer/leaky_relu_grad/SelectJgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1Lgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Egradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp=^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape?^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
�
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_3AddNEgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1*
N*
_output_shapes
:	�*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
�
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Pgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Rgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Dgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1Reshape@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Kgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeE^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
�
Sgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeL^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1L^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Ngradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulMulMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Pgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Igradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpA^gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeC^gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
�
Qgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeJ^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_4AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
gradients/AddN_5AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Kgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Sgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulL^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
�
[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulT^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Igradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_reluZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Qgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpH^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulJ^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
�
Ygradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityGgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulR^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_6AddN^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes	
:�*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;gradients/Discriminator/first_layer_1/leaky_relu_grad/zerosFill=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
9gradients/Discriminator/first_layer_1/leaky_relu_grad/SumSum<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectKgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Fgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp>^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape@^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
Ngradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeG^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
9gradients/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2ShapeYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
9gradients/Discriminator/first_layer/leaky_relu_grad/zerosFill;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
:gradients/Discriminator/first_layer/leaky_relu_grad/SelectSelect@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency9gradients/Discriminator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Select@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqual9gradients/Discriminator/first_layer/leaky_relu_grad/zerosYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
7gradients/Discriminator/first_layer/leaky_relu_grad/SumSum:gradients/Discriminator/first_layer/leaky_relu_grad/SelectIgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape7gradients/Discriminator/first_layer/leaky_relu_grad/Sum9gradients/Discriminator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Kgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Dgradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp<^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape>^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
�
Lgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeE^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_7AddN]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
N* 
_output_shapes
:
��*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
�
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Ogradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulOgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Qgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Cgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1Reshape?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Jgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeD^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
�
Rgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeK^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Ogradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Agradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Hgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp@^gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeB^gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
�
Pgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeI^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_8AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
gradients/AddN_9AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradR^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(
�
Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/noise_code_in[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d�*
transpose_a(*
transpose_b( 
�
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
�
Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1S^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
_output_shapes
:	d�*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
�
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(
�
Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_codeYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	d�*
transpose_a(
�
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
_output_shapes
:	d�*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
gradients/AddN_10AddN]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
gradients/AddN_11AddN\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1*
N*
_output_shapes
:	d�
�
beta1_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
beta2_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
�
beta2_power/readIdentitybeta2_power*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
�
eDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
�
[Discriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
UDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zerosFilleDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensor[Discriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/Const*
_output_shapes
:	d�*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0
�
CDiscriminator/first_layer/fully_connected/kernel/discriminator_opti
VariableV2*
dtype0*
_output_shapes
:	d�*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:	d�
�
JDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/AssignAssignCDiscriminator/first_layer/fully_connected/kernel/discriminator_optiUDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
HDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/readIdentityCDiscriminator/first_layer/fully_connected/kernel/discriminator_opti*
_output_shapes
:	d�*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
gDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
�
]Discriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    
�
WDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zerosFillgDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensor]Discriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
EDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1
VariableV2*
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container 
�
LDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/AssignAssignEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1WDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
JDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/readIdentityEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
SDiscriminator/first_layer/fully_connected/bias/discriminator_opti/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    
�
ADiscriminator/first_layer/fully_connected/bias/discriminator_opti
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
HDiscriminator/first_layer/fully_connected/bias/discriminator_opti/AssignAssignADiscriminator/first_layer/fully_connected/bias/discriminator_optiSDiscriminator/first_layer/fully_connected/bias/discriminator_opti/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
FDiscriminator/first_layer/fully_connected/bias/discriminator_opti/readIdentityADiscriminator/first_layer/fully_connected/bias/discriminator_opti*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:�
�
UDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
CDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:�
�
JDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/AssignAssignCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1UDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
HDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/readIdentityCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:�
�
fDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
\Discriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
VDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zerosFillfDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensor\Discriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0
�
DDiscriminator/second_layer/fully_connected/kernel/discriminator_opti
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container 
�
KDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/AssignAssignDDiscriminator/second_layer/fully_connected/kernel/discriminator_optiVDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
IDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/readIdentityDDiscriminator/second_layer/fully_connected/kernel/discriminator_opti* 
_output_shapes
:
��*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
hDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
^Discriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
XDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zerosFillhDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensor^Discriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
FDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
��
�
MDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/AssignAssignFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1XDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
KDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/readIdentityFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
TDiscriminator/second_layer/fully_connected/bias/discriminator_opti/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
BDiscriminator/second_layer/fully_connected/bias/discriminator_opti
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:�
�
IDiscriminator/second_layer/fully_connected/bias/discriminator_opti/AssignAssignBDiscriminator/second_layer/fully_connected/bias/discriminator_optiTDiscriminator/second_layer/fully_connected/bias/discriminator_opti/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
GDiscriminator/second_layer/fully_connected/bias/discriminator_opti/readIdentityBDiscriminator/second_layer/fully_connected/bias/discriminator_opti*
_output_shapes	
:�*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
VDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
DDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1
VariableV2*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
KDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/AssignAssignDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1VDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
IDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/readIdentityDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1*
_output_shapes	
:�*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
>Discriminator/prob/kernel/discriminator_opti/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
,Discriminator/prob/kernel/discriminator_opti
VariableV2*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel
�
3Discriminator/prob/kernel/discriminator_opti/AssignAssign,Discriminator/prob/kernel/discriminator_opti>Discriminator/prob/kernel/discriminator_opti/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�
�
1Discriminator/prob/kernel/discriminator_opti/readIdentity,Discriminator/prob/kernel/discriminator_opti*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
@Discriminator/prob/kernel/discriminator_opti_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
.Discriminator/prob/kernel/discriminator_opti_1
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	�
�
5Discriminator/prob/kernel/discriminator_opti_1/AssignAssign.Discriminator/prob/kernel/discriminator_opti_1@Discriminator/prob/kernel/discriminator_opti_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�
�
3Discriminator/prob/kernel/discriminator_opti_1/readIdentity.Discriminator/prob/kernel/discriminator_opti_1*
_output_shapes
:	�*
T0*,
_class"
 loc:@Discriminator/prob/kernel
�
<Discriminator/prob/bias/discriminator_opti/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
�
*Discriminator/prob/bias/discriminator_opti
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container 
�
1Discriminator/prob/bias/discriminator_opti/AssignAssign*Discriminator/prob/bias/discriminator_opti<Discriminator/prob/bias/discriminator_opti/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
�
/Discriminator/prob/bias/discriminator_opti/readIdentity*Discriminator/prob/bias/discriminator_opti*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
�
>Discriminator/prob/bias/discriminator_opti_1/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
�
,Discriminator/prob/bias/discriminator_opti_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container 
�
3Discriminator/prob/bias/discriminator_opti_1/AssignAssign,Discriminator/prob/bias/discriminator_opti_1>Discriminator/prob/bias/discriminator_opti_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
�
1Discriminator/prob/bias/discriminator_opti_1/readIdentity,Discriminator/prob/bias/discriminator_opti_1*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
e
 discriminator_opti/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *�Q9
]
discriminator_opti/beta1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
]
discriminator_opti/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
_
discriminator_opti/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Tdiscriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam0Discriminator/first_layer/fully_connected/kernelCDiscriminator/first_layer/fully_connected/kernel/discriminator_optiEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_11*
use_locking( *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	d�
�
Rdiscriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam.Discriminator/first_layer/fully_connected/biasADiscriminator/first_layer/fully_connected/bias/discriminator_optiCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_10*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
Udiscriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernelDDiscriminator/second_layer/fully_connected/kernel/discriminator_optiFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_7*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
Sdiscriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/biasBDiscriminator/second_layer/fully_connected/bias/discriminator_optiDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_6*
use_locking( *
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
=discriminator_opti/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernel,Discriminator/prob/kernel/discriminator_opti.Discriminator/prob/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_3*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
use_nesterov( *
_output_shapes
:	�*
use_locking( 
�
;discriminator_opti/update_Discriminator/prob/bias/ApplyAdam	ApplyAdamDiscriminator/prob/bias*Discriminator/prob/bias/discriminator_opti,Discriminator/prob/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_2*
use_locking( *
T0**
_class 
loc:@Discriminator/prob/bias*
use_nesterov( *
_output_shapes
:
�
discriminator_opti/mulMulbeta1_power/readdiscriminator_opti/beta1S^discriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamU^discriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam<^discriminator_opti/update_Discriminator/prob/bias/ApplyAdam>^discriminator_opti/update_Discriminator/prob/kernel/ApplyAdamT^discriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamV^discriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
�
discriminator_opti/AssignAssignbeta1_powerdiscriminator_opti/mul*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�
discriminator_opti/mul_1Mulbeta2_power/readdiscriminator_opti/beta2S^discriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamU^discriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam<^discriminator_opti/update_Discriminator/prob/bias/ApplyAdam>^discriminator_opti/update_Discriminator/prob/kernel/ApplyAdamT^discriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamV^discriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
�
discriminator_opti/Assign_1Assignbeta2_powerdiscriminator_opti/mul_1*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
�
discriminator_optiNoOp^discriminator_opti/Assign^discriminator_opti/Assign_1S^discriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamU^discriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam<^discriminator_opti/update_Discriminator/prob/bias/ApplyAdam>^discriminator_opti/update_Discriminator/prob/kernel/ApplyAdamT^discriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamV^discriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *
T0*

index_type0
K
0gradients_1/generator_loss_grad/tuple/group_depsNoOp^gradients_1/Fill
�
8gradients_1/generator_loss_grad/tuple/control_dependencyIdentitygradients_1/Fill1^gradients_1/generator_loss_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill*
_output_shapes
: 
�
:gradients_1/generator_loss_grad/tuple/control_dependency_1Identitygradients_1/Fill1^gradients_1/generator_loss_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill*
_output_shapes
: 
�
gradients_1/mul_grad/MulMul8gradients_1/generator_loss_grad/tuple/control_dependencyadversalrial_loss*
_output_shapes
: *
T0
�
gradients_1/mul_grad/Mul_1Mul8gradients_1/generator_loss_grad/tuple/control_dependencymul/x*
T0*
_output_shapes
: 
e
%gradients_1/mul_grad/tuple/group_depsNoOp^gradients_1/mul_grad/Mul^gradients_1/mul_grad/Mul_1
�
-gradients_1/mul_grad/tuple/control_dependencyIdentitygradients_1/mul_grad/Mul&^gradients_1/mul_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients_1/mul_grad/Mul*
_output_shapes
: 
�
/gradients_1/mul_grad/tuple/control_dependency_1Identitygradients_1/mul_grad/Mul_1&^gradients_1/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_grad/Mul_1*
_output_shapes
: 
�
gradients_1/mul_1_grad/MulMul:gradients_1/generator_loss_grad/tuple/control_dependency_1pixelwise_loss*
T0*
_output_shapes
: 
�
gradients_1/mul_1_grad/Mul_1Mul:gradients_1/generator_loss_grad/tuple/control_dependency_1mul_1/x*
T0*
_output_shapes
: 
k
'gradients_1/mul_1_grad/tuple/group_depsNoOp^gradients_1/mul_1_grad/Mul^gradients_1/mul_1_grad/Mul_1
�
/gradients_1/mul_1_grad/tuple/control_dependencyIdentitygradients_1/mul_1_grad/Mul(^gradients_1/mul_1_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_1_grad/Mul*
_output_shapes
: 
�
1gradients_1/mul_1_grad/tuple/control_dependency_1Identitygradients_1/mul_1_grad/Mul_1(^gradients_1/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_1_grad/Mul_1*
_output_shapes
: 
�
0gradients_1/adversalrial_loss_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
*gradients_1/adversalrial_loss_grad/ReshapeReshape/gradients_1/mul_grad/tuple/control_dependency_10gradients_1/adversalrial_loss_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
u
(gradients_1/adversalrial_loss_grad/ShapeShapelogistic_loss*
_output_shapes
:*
T0*
out_type0
�
'gradients_1/adversalrial_loss_grad/TileTile*gradients_1/adversalrial_loss_grad/Reshape(gradients_1/adversalrial_loss_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
w
*gradients_1/adversalrial_loss_grad/Shape_1Shapelogistic_loss*
T0*
out_type0*
_output_shapes
:
m
*gradients_1/adversalrial_loss_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
r
(gradients_1/adversalrial_loss_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
'gradients_1/adversalrial_loss_grad/ProdProd*gradients_1/adversalrial_loss_grad/Shape_1(gradients_1/adversalrial_loss_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
t
*gradients_1/adversalrial_loss_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
)gradients_1/adversalrial_loss_grad/Prod_1Prod*gradients_1/adversalrial_loss_grad/Shape_2*gradients_1/adversalrial_loss_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
n
,gradients_1/adversalrial_loss_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
*gradients_1/adversalrial_loss_grad/MaximumMaximum)gradients_1/adversalrial_loss_grad/Prod_1,gradients_1/adversalrial_loss_grad/Maximum/y*
T0*
_output_shapes
: 
�
+gradients_1/adversalrial_loss_grad/floordivFloorDiv'gradients_1/adversalrial_loss_grad/Prod*gradients_1/adversalrial_loss_grad/Maximum*
T0*
_output_shapes
: 
�
'gradients_1/adversalrial_loss_grad/CastCast+gradients_1/adversalrial_loss_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
*gradients_1/adversalrial_loss_grad/truedivRealDiv'gradients_1/adversalrial_loss_grad/Tile'gradients_1/adversalrial_loss_grad/Cast*'
_output_shapes
:���������*
T0
�
-gradients_1/pixelwise_loss_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
�
'gradients_1/pixelwise_loss_grad/ReshapeReshape1gradients_1/mul_1_grad/tuple/control_dependency_1-gradients_1/pixelwise_loss_grad/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:
h
%gradients_1/pixelwise_loss_grad/ShapeShapeAbs*
T0*
out_type0*
_output_shapes
:
�
$gradients_1/pixelwise_loss_grad/TileTile'gradients_1/pixelwise_loss_grad/Reshape%gradients_1/pixelwise_loss_grad/Shape*

Tmultiples0*
T0*/
_output_shapes
:���������
j
'gradients_1/pixelwise_loss_grad/Shape_1ShapeAbs*
T0*
out_type0*
_output_shapes
:
j
'gradients_1/pixelwise_loss_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
o
%gradients_1/pixelwise_loss_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
$gradients_1/pixelwise_loss_grad/ProdProd'gradients_1/pixelwise_loss_grad/Shape_1%gradients_1/pixelwise_loss_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
q
'gradients_1/pixelwise_loss_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
&gradients_1/pixelwise_loss_grad/Prod_1Prod'gradients_1/pixelwise_loss_grad/Shape_2'gradients_1/pixelwise_loss_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
k
)gradients_1/pixelwise_loss_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
'gradients_1/pixelwise_loss_grad/MaximumMaximum&gradients_1/pixelwise_loss_grad/Prod_1)gradients_1/pixelwise_loss_grad/Maximum/y*
_output_shapes
: *
T0
�
(gradients_1/pixelwise_loss_grad/floordivFloorDiv$gradients_1/pixelwise_loss_grad/Prod'gradients_1/pixelwise_loss_grad/Maximum*
_output_shapes
: *
T0
�
$gradients_1/pixelwise_loss_grad/CastCast(gradients_1/pixelwise_loss_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
'gradients_1/pixelwise_loss_grad/truedivRealDiv$gradients_1/pixelwise_loss_grad/Tile$gradients_1/pixelwise_loss_grad/Cast*
T0*/
_output_shapes
:���������
u
$gradients_1/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0*
_output_shapes
:
y
&gradients_1/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
_output_shapes
:*
T0*
out_type0
�
4gradients_1/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_1/logistic_loss_grad/Shape&gradients_1/logistic_loss_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
"gradients_1/logistic_loss_grad/SumSum*gradients_1/adversalrial_loss_grad/truediv4gradients_1/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
&gradients_1/logistic_loss_grad/ReshapeReshape"gradients_1/logistic_loss_grad/Sum$gradients_1/logistic_loss_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
$gradients_1/logistic_loss_grad/Sum_1Sum*gradients_1/adversalrial_loss_grad/truediv6gradients_1/logistic_loss_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients_1/logistic_loss_grad/Reshape_1Reshape$gradients_1/logistic_loss_grad/Sum_1&gradients_1/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/gradients_1/logistic_loss_grad/tuple/group_depsNoOp'^gradients_1/logistic_loss_grad/Reshape)^gradients_1/logistic_loss_grad/Reshape_1
�
7gradients_1/logistic_loss_grad/tuple/control_dependencyIdentity&gradients_1/logistic_loss_grad/Reshape0^gradients_1/logistic_loss_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*9
_class/
-+loc:@gradients_1/logistic_loss_grad/Reshape
�
9gradients_1/logistic_loss_grad/tuple/control_dependency_1Identity(gradients_1/logistic_loss_grad/Reshape_10^gradients_1/logistic_loss_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_grad/Reshape_1*'
_output_shapes
:���������
`
gradients_1/Abs_grad/SignSignSub*/
_output_shapes
:���������*
T0
�
gradients_1/Abs_grad/mulMul'gradients_1/pixelwise_loss_grad/truedivgradients_1/Abs_grad/Sign*/
_output_shapes
:���������*
T0
|
(gradients_1/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0*
_output_shapes
:
{
*gradients_1/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
_output_shapes
:*
T0*
out_type0
�
8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/sub_grad/Shape*gradients_1/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients_1/logistic_loss/sub_grad/SumSum7gradients_1/logistic_loss_grad/tuple/control_dependency8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients_1/logistic_loss/sub_grad/ReshapeReshape&gradients_1/logistic_loss/sub_grad/Sum(gradients_1/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients_1/logistic_loss/sub_grad/Sum_1Sum7gradients_1/logistic_loss_grad/tuple/control_dependency:gradients_1/logistic_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
z
&gradients_1/logistic_loss/sub_grad/NegNeg(gradients_1/logistic_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
�
,gradients_1/logistic_loss/sub_grad/Reshape_1Reshape&gradients_1/logistic_loss/sub_grad/Neg*gradients_1/logistic_loss/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
3gradients_1/logistic_loss/sub_grad/tuple/group_depsNoOp+^gradients_1/logistic_loss/sub_grad/Reshape-^gradients_1/logistic_loss/sub_grad/Reshape_1
�
;gradients_1/logistic_loss/sub_grad/tuple/control_dependencyIdentity*gradients_1/logistic_loss/sub_grad/Reshape4^gradients_1/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss/sub_grad/Reshape*'
_output_shapes
:���������
�
=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1Identity,gradients_1/logistic_loss/sub_grad/Reshape_14^gradients_1/logistic_loss/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:���������
�
*gradients_1/logistic_loss/Log1p_grad/add/xConst:^gradients_1/logistic_loss_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
(gradients_1/logistic_loss/Log1p_grad/addAdd*gradients_1/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:���������
�
/gradients_1/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_1/logistic_loss/Log1p_grad/add*'
_output_shapes
:���������*
T0
�
(gradients_1/logistic_loss/Log1p_grad/mulMul9gradients_1/logistic_loss_grad/tuple/control_dependency_1/gradients_1/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:���������*
T0
o
gradients_1/Sub_grad/ShapeShapeDecoder/reshape_image*
_output_shapes
:*
T0*
out_type0
k
gradients_1/Sub_grad/Shape_1ShapeEncoder/real_in*
T0*
out_type0*
_output_shapes
:
�
*gradients_1/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Sub_grad/Shapegradients_1/Sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/Sub_grad/SumSumgradients_1/Abs_grad/mul*gradients_1/Sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients_1/Sub_grad/ReshapeReshapegradients_1/Sub_grad/Sumgradients_1/Sub_grad/Shape*
T0*
Tshape0*/
_output_shapes
:���������
�
gradients_1/Sub_grad/Sum_1Sumgradients_1/Abs_grad/mul,gradients_1/Sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
^
gradients_1/Sub_grad/NegNeggradients_1/Sub_grad/Sum_1*
T0*
_output_shapes
:
�
gradients_1/Sub_grad/Reshape_1Reshapegradients_1/Sub_grad/Neggradients_1/Sub_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:���������
m
%gradients_1/Sub_grad/tuple/group_depsNoOp^gradients_1/Sub_grad/Reshape^gradients_1/Sub_grad/Reshape_1
�
-gradients_1/Sub_grad/tuple/control_dependencyIdentitygradients_1/Sub_grad/Reshape&^gradients_1/Sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/Sub_grad/Reshape*/
_output_shapes
:���������
�
/gradients_1/Sub_grad/tuple/control_dependency_1Identitygradients_1/Sub_grad/Reshape_1&^gradients_1/Sub_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/Sub_grad/Reshape_1*/
_output_shapes
:���������
�
0gradients_1/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:���������*
T0
�
,gradients_1/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual;gradients_1/logistic_loss/sub_grad/tuple/control_dependency0gradients_1/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:���������
�
.gradients_1/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_1/logistic_loss/Select_grad/zeros_like;gradients_1/logistic_loss/sub_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
6gradients_1/logistic_loss/Select_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss/Select_grad/Select/^gradients_1/logistic_loss/Select_grad/Select_1
�
>gradients_1/logistic_loss/Select_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss/Select_grad/Select7^gradients_1/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select*'
_output_shapes
:���������
�
@gradients_1/logistic_loss/Select_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss/Select_grad/Select_17^gradients_1/logistic_loss/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss/Select_grad/Select_1*'
_output_shapes
:���������
�
(gradients_1/logistic_loss/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
_output_shapes
:*
T0*
out_type0
s
*gradients_1/logistic_loss/mul_grad/Shape_1Shape	ones_like*
T0*
out_type0*
_output_shapes
:
�
8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/mul_grad/Shape*gradients_1/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients_1/logistic_loss/mul_grad/MulMul=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*
T0*'
_output_shapes
:���������
�
&gradients_1/logistic_loss/mul_grad/SumSum&gradients_1/logistic_loss/mul_grad/Mul8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*gradients_1/logistic_loss/mul_grad/ReshapeReshape&gradients_1/logistic_loss/mul_grad/Sum(gradients_1/logistic_loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
(gradients_1/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
(gradients_1/logistic_loss/mul_grad/Sum_1Sum(gradients_1/logistic_loss/mul_grad/Mul_1:gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
,gradients_1/logistic_loss/mul_grad/Reshape_1Reshape(gradients_1/logistic_loss/mul_grad/Sum_1*gradients_1/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
3gradients_1/logistic_loss/mul_grad/tuple/group_depsNoOp+^gradients_1/logistic_loss/mul_grad/Reshape-^gradients_1/logistic_loss/mul_grad/Reshape_1
�
;gradients_1/logistic_loss/mul_grad/tuple/control_dependencyIdentity*gradients_1/logistic_loss/mul_grad/Reshape4^gradients_1/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss/mul_grad/Reshape*'
_output_shapes
:���������
�
=gradients_1/logistic_loss/mul_grad/tuple/control_dependency_1Identity,gradients_1/logistic_loss/mul_grad/Reshape_14^gradients_1/logistic_loss/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:���������
�
&gradients_1/logistic_loss/Exp_grad/mulMul(gradients_1/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0*'
_output_shapes
:���������
~
,gradients_1/Decoder/reshape_image_grad/ShapeShapeDecoder/last_layer*
T0*
out_type0*
_output_shapes
:
�
.gradients_1/Decoder/reshape_image_grad/ReshapeReshape-gradients_1/Sub_grad/tuple/control_dependency,gradients_1/Decoder/reshape_image_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
2gradients_1/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:���������
�
.gradients_1/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_1/logistic_loss/Exp_grad/mul2gradients_1/logistic_loss/Select_1_grad/zeros_like*
T0*'
_output_shapes
:���������
�
0gradients_1/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_1/logistic_loss/Select_1_grad/zeros_like&gradients_1/logistic_loss/Exp_grad/mul*'
_output_shapes
:���������*
T0
�
8gradients_1/logistic_loss/Select_1_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss/Select_1_grad/Select1^gradients_1/logistic_loss/Select_1_grad/Select_1
�
@gradients_1/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss/Select_1_grad/Select9^gradients_1/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss/Select_1_grad/Select*'
_output_shapes
:���������
�
Bgradients_1/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss/Select_1_grad/Select_19^gradients_1/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*C
_class9
75loc:@gradients_1/logistic_loss/Select_1_grad/Select_1
�
,gradients_1/Decoder/last_layer_grad/TanhGradTanhGradDecoder/last_layer.gradients_1/Decoder/reshape_image_grad/Reshape*
T0*(
_output_shapes
:����������
�
&gradients_1/logistic_loss/Neg_grad/NegNeg@gradients_1/logistic_loss/Select_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
2gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/Decoder/last_layer_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
�
7gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad-^gradients_1/Decoder/last_layer_grad/TanhGrad
�
?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_1/Decoder/last_layer_grad/TanhGrad8^gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*?
_class5
31loc:@gradients_1/Decoder/last_layer_grad/TanhGrad
�
Agradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad8^gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
,gradients_1/Decoder/dense/MatMul_grad/MatMulMatMul?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependencyDecoder/dense/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
.gradients_1/Decoder/dense/MatMul_grad/MatMul_1MatMulDecoder/second_layer/leaky_relu?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
6gradients_1/Decoder/dense/MatMul_grad/tuple/group_depsNoOp-^gradients_1/Decoder/dense/MatMul_grad/MatMul/^gradients_1/Decoder/dense/MatMul_grad/MatMul_1
�
>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/Decoder/dense/MatMul_grad/MatMul7^gradients_1/Decoder/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/Decoder/dense/MatMul_grad/MatMul*(
_output_shapes
:����������
�
@gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/Decoder/dense/MatMul_grad/MatMul_17^gradients_1/Decoder/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/Decoder/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddNAddN>gradients_1/logistic_loss/Select_grad/tuple/control_dependency;gradients_1/logistic_loss/mul_grad/tuple/control_dependencyBgradients_1/logistic_loss/Select_1_grad/tuple/control_dependency_1&gradients_1/logistic_loss/Neg_grad/Neg*
N*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select
�
7gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes
:
�
<gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN8^gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad
�
Dgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN=^gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select
�
Fgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity7gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad=^gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
6gradients_1/Decoder/second_layer/leaky_relu_grad/ShapeShape#Decoder/second_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1Shape8Decoder/second_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_2Shape>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
<gradients_1/Decoder/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6gradients_1/Decoder/second_layer/leaky_relu_grad/zerosFill8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_2<gradients_1/Decoder/second_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Decoder/second_layer/leaky_relu/mul8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Fgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Decoder/second_layer/leaky_relu_grad/Shape8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients_1/Decoder/second_layer/leaky_relu_grad/SelectSelect=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqual>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency6gradients_1/Decoder/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
9gradients_1/Decoder/second_layer/leaky_relu_grad/Select_1Select=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqual6gradients_1/Decoder/second_layer/leaky_relu_grad/zeros>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
4gradients_1/Decoder/second_layer/leaky_relu_grad/SumSum7gradients_1/Decoder/second_layer/leaky_relu_grad/SelectFgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
8gradients_1/Decoder/second_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Decoder/second_layer/leaky_relu_grad/Sum6gradients_1/Decoder/second_layer/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
6gradients_1/Decoder/second_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Decoder/second_layer/leaky_relu_grad/Select_1Hgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Decoder/second_layer/leaky_relu_grad/Sum_18gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Agradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape;^gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1
�
Igradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Decoder/second_layer/leaky_relu_grad/ReshapeB^gradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Kgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1B^gradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*M
_classC
A?loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1
�
1gradients_1/Discriminator/prob/MatMul_grad/MatMulMatMulDgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
3gradients_1/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluDgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
;gradients_1/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp2^gradients_1/Discriminator/prob/MatMul_grad/MatMul4^gradients_1/Discriminator/prob/MatMul_grad/MatMul_1
�
Cgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity1gradients_1/Discriminator/prob/MatMul_grad/MatMul<^gradients_1/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/Discriminator/prob/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Egradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity3gradients_1/Discriminator/prob/MatMul_grad/MatMul_1<^gradients_1/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	�
}
:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1Shape8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Jgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/MulMulIgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency8Decoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/SumSum8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/MulJgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Mul_1Mul%Decoder/second_layer/leaky_relu/alphaIgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Egradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1
�
Mgradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*O
_classE
CAloc:@gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape
�
Ogradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
<gradients_1/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<gradients_1/Discriminator/second_layer/leaky_relu_grad/zerosFill>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_2Bgradients_1/Discriminator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Cgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Lgradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
=gradients_1/Discriminator/second_layer/leaky_relu_grad/SelectSelectCgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqualCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency<gradients_1/Discriminator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
?gradients_1/Discriminator/second_layer/leaky_relu_grad/Select_1SelectCgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqual<gradients_1/Discriminator/second_layer/leaky_relu_grad/zerosCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
:gradients_1/Discriminator/second_layer/leaky_relu_grad/SumSum=gradients_1/Discriminator/second_layer/leaky_relu_grad/SelectLgradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape:gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum<gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
<gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum?gradients_1/Discriminator/second_layer/leaky_relu_grad/Select_1Ngradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape<gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum_1>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Ggradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp?^gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeA^gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1
�
Ogradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity>gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeH^gradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Qgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1H^gradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_1AddNKgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*M
_classC
A?loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Decoder/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_1_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_1agradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Sgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape
�
dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Pgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ShapeBgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/MulMulOgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/SumSum>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/MulPgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaOgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Rgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Dgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Kgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
�
Sgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeL^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ugradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1L^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Decoder/second_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Decoder/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Decoder/second_layer/fully_connected/BiasAddbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Kgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
Xgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg
�
`gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
gradients_1/AddN_2AddNQgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Ugradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Tgradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2P^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2U^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradU^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Igradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Ngradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Vgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
Xgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Decoder/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Decoder/second_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:�
�
Zgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
�
dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Igradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMul\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Kgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_relu\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Sgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulL^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
�
[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulT^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
]gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1T^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
�
Cgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Decoder/second_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Egradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1MatMulDecoder/first_layer/leaky_reluVgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
�
Mgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1
�
Ugradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Wgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_3AddNdgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Kgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_33Decoder/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_38Decoder/second_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
Xgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
`gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*^
_classT
RPloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul
�
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*`
_classV
TRloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
;gradients_1/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_2Shape[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Agradients_1/Discriminator/first_layer/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
;gradients_1/Discriminator/first_layer/leaky_relu_grad/zerosFill=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_2Agradients_1/Discriminator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Bgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Kgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients_1/Discriminator/first_layer/leaky_relu_grad/SelectSelectBgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqual[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency;gradients_1/Discriminator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
>gradients_1/Discriminator/first_layer/leaky_relu_grad/Select_1SelectBgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqual;gradients_1/Discriminator/first_layer/leaky_relu_grad/zeros[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
9gradients_1/Discriminator/first_layer/leaky_relu_grad/SumSum<gradients_1/Discriminator/first_layer/leaky_relu_grad/SelectKgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
=gradients_1/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape9gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum;gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
;gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum>gradients_1/Discriminator/first_layer/leaky_relu_grad/Select_1Mgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
?gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape;gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum_1=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Fgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp>^gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape@^gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1
�
Ngradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity=gradients_1/Discriminator/first_layer/leaky_relu_grad/ReshapeG^gradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Pgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity?gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1G^gradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
5gradients_1/Decoder/first_layer/leaky_relu_grad/ShapeShape"Decoder/first_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1Shape+Decoder/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
;gradients_1/Decoder/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
5gradients_1/Decoder/first_layer/leaky_relu_grad/zerosFill7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_2;gradients_1/Decoder/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual"Decoder/first_layer/leaky_relu/mul+Decoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Egradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/Decoder/first_layer/leaky_relu_grad/Shape7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6gradients_1/Decoder/first_layer/leaky_relu_grad/SelectSelect<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqualUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency5gradients_1/Decoder/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
8gradients_1/Decoder/first_layer/leaky_relu_grad/Select_1Select<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqual5gradients_1/Decoder/first_layer/leaky_relu_grad/zerosUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
3gradients_1/Decoder/first_layer/leaky_relu_grad/SumSum6gradients_1/Decoder/first_layer/leaky_relu_grad/SelectEgradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
7gradients_1/Decoder/first_layer/leaky_relu_grad/ReshapeReshape3gradients_1/Decoder/first_layer/leaky_relu_grad/Sum5gradients_1/Decoder/first_layer/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
5gradients_1/Decoder/first_layer/leaky_relu_grad/Sum_1Sum8gradients_1/Decoder/first_layer/leaky_relu_grad/Select_1Ggradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
9gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1Reshape5gradients_1/Decoder/first_layer/leaky_relu_grad/Sum_17gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
@gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_depsNoOp8^gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape:^gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1
�
Hgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity7gradients_1/Decoder/first_layer/leaky_relu_grad/ReshapeA^gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Jgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity9gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1A^gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1
�
?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Ogradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ShapeAgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/MulMulNgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/SumSum=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/MulOgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaNgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Qgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Cgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Jgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeD^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
�
Rgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeK^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Tgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1K^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*V
_classL
JHloc:@gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
|
9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1Shape+Decoder/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Igradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/MulMulHgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency+Decoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/SumSum7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/MulIgradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ReshapeReshape7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Mul_1Mul$Decoder/first_layer/leaky_relu/alphaHgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum_1Sum9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Mul_1Kgradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
=gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1Reshape9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum_1;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Dgradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp<^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape>^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1
�
Lgradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ngradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity=gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1E^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_4AddNPgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Tgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1
�
Ngradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Sgradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_4O^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_4T^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1
�
]gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradT^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
gradients_1/AddN_5AddNJgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Ngradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Hgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Mgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_5I^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ugradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_5N^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1
�
Wgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradN^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Hgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMul[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
�
Jgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_code[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	d�*
transpose_a(
�
Rgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulK^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Zgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulS^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
\gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1S^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d�
�
Bgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulMatMulUgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency/Decoder/first_layer/fully_connected/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
�
Dgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_codeUgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d�*
transpose_a(*
transpose_b( *
T0
�
Lgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpC^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulE^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Tgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityBgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulM^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
Vgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityDgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1M^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d�
�
gradients_1/AddN_6AddNZgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyTgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*[
_classQ
OMloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*
N*'
_output_shapes
:���������d
�
1gradients_1/Encoder/encoder_code_grad/SigmoidGradSigmoidGradEncoder/encoder_codegradients_1/AddN_6*
T0*'
_output_shapes
:���������d
t
"gradients_1/Encoder/Add_grad/ShapeShapeEncoder/logvar_std*
_output_shapes
:*
T0*
out_type0
~
$gradients_1/Encoder/Add_grad/Shape_1ShapeEncoder/encoder_mu/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
2gradients_1/Encoder/Add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_1/Encoder/Add_grad/Shape$gradients_1/Encoder/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
 gradients_1/Encoder/Add_grad/SumSum1gradients_1/Encoder/encoder_code_grad/SigmoidGrad2gradients_1/Encoder/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
$gradients_1/Encoder/Add_grad/ReshapeReshape gradients_1/Encoder/Add_grad/Sum"gradients_1/Encoder/Add_grad/Shape*'
_output_shapes
:���������d*
T0*
Tshape0
�
"gradients_1/Encoder/Add_grad/Sum_1Sum1gradients_1/Encoder/encoder_code_grad/SigmoidGrad4gradients_1/Encoder/Add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&gradients_1/Encoder/Add_grad/Reshape_1Reshape"gradients_1/Encoder/Add_grad/Sum_1$gradients_1/Encoder/Add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������d
�
-gradients_1/Encoder/Add_grad/tuple/group_depsNoOp%^gradients_1/Encoder/Add_grad/Reshape'^gradients_1/Encoder/Add_grad/Reshape_1
�
5gradients_1/Encoder/Add_grad/tuple/control_dependencyIdentity$gradients_1/Encoder/Add_grad/Reshape.^gradients_1/Encoder/Add_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*7
_class-
+)loc:@gradients_1/Encoder/Add_grad/Reshape
�
7gradients_1/Encoder/Add_grad/tuple/control_dependency_1Identity&gradients_1/Encoder/Add_grad/Reshape_1.^gradients_1/Encoder/Add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/Encoder/Add_grad/Reshape_1*'
_output_shapes
:���������d
s
)gradients_1/Encoder/logvar_std_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:
v
+gradients_1/Encoder/logvar_std_grad/Shape_1ShapeEncoder/Exp*
_output_shapes
:*
T0*
out_type0
�
9gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients_1/Encoder/logvar_std_grad/Shape+gradients_1/Encoder/logvar_std_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
'gradients_1/Encoder/logvar_std_grad/MulMul5gradients_1/Encoder/Add_grad/tuple/control_dependencyEncoder/Exp*
T0*'
_output_shapes
:���������d
�
'gradients_1/Encoder/logvar_std_grad/SumSum'gradients_1/Encoder/logvar_std_grad/Mul9gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
+gradients_1/Encoder/logvar_std_grad/ReshapeReshape'gradients_1/Encoder/logvar_std_grad/Sum)gradients_1/Encoder/logvar_std_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
�
)gradients_1/Encoder/logvar_std_grad/Mul_1MulEncoder/random_normal5gradients_1/Encoder/Add_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������d
�
)gradients_1/Encoder/logvar_std_grad/Sum_1Sum)gradients_1/Encoder/logvar_std_grad/Mul_1;gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
-gradients_1/Encoder/logvar_std_grad/Reshape_1Reshape)gradients_1/Encoder/logvar_std_grad/Sum_1+gradients_1/Encoder/logvar_std_grad/Shape_1*'
_output_shapes
:���������d*
T0*
Tshape0
�
4gradients_1/Encoder/logvar_std_grad/tuple/group_depsNoOp,^gradients_1/Encoder/logvar_std_grad/Reshape.^gradients_1/Encoder/logvar_std_grad/Reshape_1
�
<gradients_1/Encoder/logvar_std_grad/tuple/control_dependencyIdentity+gradients_1/Encoder/logvar_std_grad/Reshape5^gradients_1/Encoder/logvar_std_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_1/Encoder/logvar_std_grad/Reshape*
_output_shapes
:d
�
>gradients_1/Encoder/logvar_std_grad/tuple/control_dependency_1Identity-gradients_1/Encoder/logvar_std_grad/Reshape_15^gradients_1/Encoder/logvar_std_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/Encoder/logvar_std_grad/Reshape_1*'
_output_shapes
:���������d
�
7gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/Encoder/Add_grad/tuple/control_dependency_1*
T0*
data_formatNHWC*
_output_shapes
:d
�
<gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_depsNoOp8^gradients_1/Encoder/Add_grad/tuple/control_dependency_18^gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad
�
Dgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/Encoder/Add_grad/tuple/control_dependency_1=^gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/Encoder/Add_grad/Reshape_1*'
_output_shapes
:���������d
�
Fgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency_1Identity7gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad=^gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d
�
 gradients_1/Encoder/Exp_grad/mulMul>gradients_1/Encoder/logvar_std_grad/tuple/control_dependency_1Encoder/Exp*'
_output_shapes
:���������d*
T0
�
1gradients_1/Encoder/encoder_mu/MatMul_grad/MatMulMatMulDgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependencyEncoder/encoder_mu/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
3gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1MatMulEncoder/second_layer/leaky_reluDgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	�d*
transpose_a(
�
;gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_depsNoOp2^gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul4^gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1
�
Cgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependencyIdentity1gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul<^gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Egradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependency_1Identity3gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1<^gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1*
_output_shapes
:	�d
�
&gradients_1/Encoder/truediv_grad/ShapeShapeEncoder/encoder_logvar/BiasAdd*
T0*
out_type0*
_output_shapes
:
k
(gradients_1/Encoder/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
6gradients_1/Encoder/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/Encoder/truediv_grad/Shape(gradients_1/Encoder/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
(gradients_1/Encoder/truediv_grad/RealDivRealDiv gradients_1/Encoder/Exp_grad/mulEncoder/truediv/y*
T0*'
_output_shapes
:���������d
�
$gradients_1/Encoder/truediv_grad/SumSum(gradients_1/Encoder/truediv_grad/RealDiv6gradients_1/Encoder/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(gradients_1/Encoder/truediv_grad/ReshapeReshape$gradients_1/Encoder/truediv_grad/Sum&gradients_1/Encoder/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������d
}
$gradients_1/Encoder/truediv_grad/NegNegEncoder/encoder_logvar/BiasAdd*
T0*'
_output_shapes
:���������d
�
*gradients_1/Encoder/truediv_grad/RealDiv_1RealDiv$gradients_1/Encoder/truediv_grad/NegEncoder/truediv/y*
T0*'
_output_shapes
:���������d
�
*gradients_1/Encoder/truediv_grad/RealDiv_2RealDiv*gradients_1/Encoder/truediv_grad/RealDiv_1Encoder/truediv/y*'
_output_shapes
:���������d*
T0
�
$gradients_1/Encoder/truediv_grad/mulMul gradients_1/Encoder/Exp_grad/mul*gradients_1/Encoder/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:���������d
�
&gradients_1/Encoder/truediv_grad/Sum_1Sum$gradients_1/Encoder/truediv_grad/mul8gradients_1/Encoder/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients_1/Encoder/truediv_grad/Reshape_1Reshape&gradients_1/Encoder/truediv_grad/Sum_1(gradients_1/Encoder/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
1gradients_1/Encoder/truediv_grad/tuple/group_depsNoOp)^gradients_1/Encoder/truediv_grad/Reshape+^gradients_1/Encoder/truediv_grad/Reshape_1
�
9gradients_1/Encoder/truediv_grad/tuple/control_dependencyIdentity(gradients_1/Encoder/truediv_grad/Reshape2^gradients_1/Encoder/truediv_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/Encoder/truediv_grad/Reshape*'
_output_shapes
:���������d
�
;gradients_1/Encoder/truediv_grad/tuple/control_dependency_1Identity*gradients_1/Encoder/truediv_grad/Reshape_12^gradients_1/Encoder/truediv_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/Encoder/truediv_grad/Reshape_1*
_output_shapes
: 
�
;gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients_1/Encoder/truediv_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:d
�
@gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_depsNoOp<^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGrad:^gradients_1/Encoder/truediv_grad/tuple/control_dependency
�
Hgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependencyIdentity9gradients_1/Encoder/truediv_grad/tuple/control_dependencyA^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*;
_class1
/-loc:@gradients_1/Encoder/truediv_grad/Reshape
�
Jgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGradA^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d
�
5gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMulMatMulHgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency"Encoder/encoder_logvar/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
7gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1MatMulEncoder/second_layer/leaky_reluHgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�d*
transpose_a(*
transpose_b( 
�
?gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_depsNoOp6^gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul8^gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1
�
Ggradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependencyIdentity5gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul@^gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*H
_class>
<:loc:@gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul
�
Igradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency_1Identity7gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1@^gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1*
_output_shapes
:	�d
�
gradients_1/AddN_7AddNCgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependencyGgradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency*
N*(
_output_shapes
:����������*
T0*D
_class:
86loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul
�
6gradients_1/Encoder/second_layer/leaky_relu_grad/ShapeShape#Encoder/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1Shape8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_2Shapegradients_1/AddN_7*
T0*
out_type0*
_output_shapes
:
�
<gradients_1/Encoder/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6gradients_1/Encoder/second_layer/leaky_relu_grad/zerosFill8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_2<gradients_1/Encoder/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Encoder/second_layer/leaky_relu/mul8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Fgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Encoder/second_layer/leaky_relu_grad/Shape8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients_1/Encoder/second_layer/leaky_relu_grad/SelectSelect=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqualgradients_1/AddN_76gradients_1/Encoder/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
9gradients_1/Encoder/second_layer/leaky_relu_grad/Select_1Select=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqual6gradients_1/Encoder/second_layer/leaky_relu_grad/zerosgradients_1/AddN_7*
T0*(
_output_shapes
:����������
�
4gradients_1/Encoder/second_layer/leaky_relu_grad/SumSum7gradients_1/Encoder/second_layer/leaky_relu_grad/SelectFgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
8gradients_1/Encoder/second_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Encoder/second_layer/leaky_relu_grad/Sum6gradients_1/Encoder/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
6gradients_1/Encoder/second_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Encoder/second_layer/leaky_relu_grad/Select_1Hgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Encoder/second_layer/leaky_relu_grad/Sum_18gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Agradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape;^gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1
�
Igradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Encoder/second_layer/leaky_relu_grad/ReshapeB^gradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Kgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1B^gradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*M
_classC
A?loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1
}
:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1Shape8Encoder/second_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
Jgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/MulMulIgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/SumSum8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/MulJgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Mul_1Mul%Encoder/second_layer/leaky_relu/alphaIgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Egradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1
�
Mgradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ogradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_8AddNKgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Encoder/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_8_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_8agradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Encoder/second_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Encoder/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
�
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Encoder/second_layer/fully_connected/BiasAddbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Kgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
Xgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg
�
`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Igradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Ngradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Vgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
Xgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Encoder/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Encoder/second_layer/batch_normalization/moving_mean/read*
_output_shapes	
:�*
T0
�
Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
�
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Cgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Encoder/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Egradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/first_layer/leaky_reluVgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Mgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1
�
Ugradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Wgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*X
_classN
LJloc:@gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1
�
gradients_1/AddN_9AddNdgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Kgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_93Encoder/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_98Encoder/second_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
Xgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*^
_classT
RPloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul
�
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*`
_classV
TRloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
5gradients_1/Encoder/first_layer/leaky_relu_grad/ShapeShape"Encoder/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1Shape+Encoder/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
;gradients_1/Encoder/first_layer/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
5gradients_1/Encoder/first_layer/leaky_relu_grad/zerosFill7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_2;gradients_1/Encoder/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual"Encoder/first_layer/leaky_relu/mul+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Egradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/Encoder/first_layer/leaky_relu_grad/Shape7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
6gradients_1/Encoder/first_layer/leaky_relu_grad/SelectSelect<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqualUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency5gradients_1/Encoder/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
8gradients_1/Encoder/first_layer/leaky_relu_grad/Select_1Select<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqual5gradients_1/Encoder/first_layer/leaky_relu_grad/zerosUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
3gradients_1/Encoder/first_layer/leaky_relu_grad/SumSum6gradients_1/Encoder/first_layer/leaky_relu_grad/SelectEgradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
7gradients_1/Encoder/first_layer/leaky_relu_grad/ReshapeReshape3gradients_1/Encoder/first_layer/leaky_relu_grad/Sum5gradients_1/Encoder/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
5gradients_1/Encoder/first_layer/leaky_relu_grad/Sum_1Sum8gradients_1/Encoder/first_layer/leaky_relu_grad/Select_1Ggradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
9gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1Reshape5gradients_1/Encoder/first_layer/leaky_relu_grad/Sum_17gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
@gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_depsNoOp8^gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape:^gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1
�
Hgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity7gradients_1/Encoder/first_layer/leaky_relu_grad/ReshapeA^gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Jgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity9gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1A^gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
|
9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1Shape+Encoder/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Igradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/MulMulHgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/SumSum7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/MulIgradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ReshapeReshape7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Mul_1Mul$Encoder/first_layer/leaky_relu/alphaHgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum_1Sum9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Mul_1Kgradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1Reshape9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum_1;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Dgradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp<^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape>^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1
�
Lgradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*N
_classD
B@loc:@gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape
�
Ngradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity=gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1E^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*P
_classF
DBloc:@gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1
�
gradients_1/AddN_10AddNJgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Ngradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Hgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_10*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Mgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_10I^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ugradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_10N^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1
�
Wgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradN^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Bgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulMatMulUgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency/Encoder/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Dgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/ReshapeUgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Lgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpC^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulE^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Tgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityBgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulM^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*U
_classK
IGloc:@gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul
�
Vgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityDgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1M^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
beta1_power_1/initial_valueConst*%
_class
loc:@Decoder/dense/bias*
valueB
 *   ?*
dtype0*
_output_shapes
: 
�
beta1_power_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@Decoder/dense/bias
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes
: 
u
beta1_power_1/readIdentitybeta1_power_1*
_output_shapes
: *
T0*%
_class
loc:@Decoder/dense/bias
�
beta2_power_1/initial_valueConst*%
_class
loc:@Decoder/dense/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@Decoder/dense/bias
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes
: 
u
beta2_power_1/readIdentitybeta2_power_1*
_output_shapes
: *
T0*%
_class
loc:@Decoder/dense/bias
�
[Encoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
QEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
KEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill[Encoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorQEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
9Encoder/first_layer/fully_connected/kernel/generator_opti
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container 
�
@Encoder/first_layer/fully_connected/kernel/generator_opti/AssignAssign9Encoder/first_layer/fully_connected/kernel/generator_optiKEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
>Encoder/first_layer/fully_connected/kernel/generator_opti/readIdentity9Encoder/first_layer/fully_connected/kernel/generator_opti*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
]Encoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
SEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
MEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill]Encoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorSEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
;Encoder/first_layer/fully_connected/kernel/generator_opti_1
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container 
�
BEncoder/first_layer/fully_connected/kernel/generator_opti_1/AssignAssign;Encoder/first_layer/fully_connected/kernel/generator_opti_1MEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
@Encoder/first_layer/fully_connected/kernel/generator_opti_1/readIdentity;Encoder/first_layer/fully_connected/kernel/generator_opti_1* 
_output_shapes
:
��*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel
�
IEncoder/first_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7Encoder/first_layer/fully_connected/bias/generator_opti
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
	container *
shape:�
�
>Encoder/first_layer/fully_connected/bias/generator_opti/AssignAssign7Encoder/first_layer/fully_connected/bias/generator_optiIEncoder/first_layer/fully_connected/bias/generator_opti/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
<Encoder/first_layer/fully_connected/bias/generator_opti/readIdentity7Encoder/first_layer/fully_connected/bias/generator_opti*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:�
�
KEncoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
9Encoder/first_layer/fully_connected/bias/generator_opti_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias
�
@Encoder/first_layer/fully_connected/bias/generator_opti_1/AssignAssign9Encoder/first_layer/fully_connected/bias/generator_opti_1KEncoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
>Encoder/first_layer/fully_connected/bias/generator_opti_1/readIdentity9Encoder/first_layer/fully_connected/bias/generator_opti_1*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:�
�
\Encoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      
�
REncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LEncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill\Encoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorREncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
:Encoder/second_layer/fully_connected/kernel/generator_opti
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:
��
�
AEncoder/second_layer/fully_connected/kernel/generator_opti/AssignAssign:Encoder/second_layer/fully_connected/kernel/generator_optiLEncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
?Encoder/second_layer/fully_connected/kernel/generator_opti/readIdentity:Encoder/second_layer/fully_connected/kernel/generator_opti*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
^Encoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
TEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill^Encoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorTEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*

index_type0
�
<Encoder/second_layer/fully_connected/kernel/generator_opti_1
VariableV2*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
CEncoder/second_layer/fully_connected/kernel/generator_opti_1/AssignAssign<Encoder/second_layer/fully_connected/kernel/generator_opti_1NEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
�
AEncoder/second_layer/fully_connected/kernel/generator_opti_1/readIdentity<Encoder/second_layer/fully_connected/kernel/generator_opti_1* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
�
JEncoder/second_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
8Encoder/second_layer/fully_connected/bias/generator_opti
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container 
�
?Encoder/second_layer/fully_connected/bias/generator_opti/AssignAssign8Encoder/second_layer/fully_connected/bias/generator_optiJEncoder/second_layer/fully_connected/bias/generator_opti/Initializer/zeros*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
=Encoder/second_layer/fully_connected/bias/generator_opti/readIdentity8Encoder/second_layer/fully_connected/bias/generator_opti*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
_output_shapes	
:�
�
LEncoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
:Encoder/second_layer/fully_connected/bias/generator_opti_1
VariableV2*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
AEncoder/second_layer/fully_connected/bias/generator_opti_1/AssignAssign:Encoder/second_layer/fully_connected/bias/generator_opti_1LEncoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias
�
?Encoder/second_layer/fully_connected/bias/generator_opti_1/readIdentity:Encoder/second_layer/fully_connected/bias/generator_opti_1*
_output_shapes	
:�*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias
�
OEncoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zerosConst*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
=Encoder/second_layer/batch_normalization/gamma/generator_opti
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container 
�
DEncoder/second_layer/batch_normalization/gamma/generator_opti/AssignAssign=Encoder/second_layer/batch_normalization/gamma/generator_optiOEncoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zeros*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
BEncoder/second_layer/batch_normalization/gamma/generator_opti/readIdentity=Encoder/second_layer/batch_normalization/gamma/generator_opti*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
QEncoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB�*    
�
?Encoder/second_layer/batch_normalization/gamma/generator_opti_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container *
shape:�
�
FEncoder/second_layer/batch_normalization/gamma/generator_opti_1/AssignAssign?Encoder/second_layer/batch_normalization/gamma/generator_opti_1QEncoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
DEncoder/second_layer/batch_normalization/gamma/generator_opti_1/readIdentity?Encoder/second_layer/batch_normalization/gamma/generator_opti_1*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
NEncoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zerosConst*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
<Encoder/second_layer/batch_normalization/beta/generator_opti
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta
�
CEncoder/second_layer/batch_normalization/beta/generator_opti/AssignAssign<Encoder/second_layer/batch_normalization/beta/generator_optiNEncoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta
�
AEncoder/second_layer/batch_normalization/beta/generator_opti/readIdentity<Encoder/second_layer/batch_normalization/beta/generator_opti*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
PEncoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB�*    
�
>Encoder/second_layer/batch_normalization/beta/generator_opti_1
VariableV2*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
EEncoder/second_layer/batch_normalization/beta/generator_opti_1/AssignAssign>Encoder/second_layer/batch_normalization/beta/generator_opti_1PEncoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zeros*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
CEncoder/second_layer/batch_normalization/beta/generator_opti_1/readIdentity>Encoder/second_layer/batch_normalization/beta/generator_opti_1*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
JEncoder/encoder_mu/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
�
@Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *    
�
:Encoder/encoder_mu/kernel/generator_opti/Initializer/zerosFillJEncoder/encoder_mu/kernel/generator_opti/Initializer/zeros/shape_as_tensor@Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros/Const*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*

index_type0*
_output_shapes
:	�d
�
(Encoder/encoder_mu/kernel/generator_opti
VariableV2*
dtype0*
_output_shapes
:	�d*
shared_name *,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container *
shape:	�d
�
/Encoder/encoder_mu/kernel/generator_opti/AssignAssign(Encoder/encoder_mu/kernel/generator_opti:Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	�d
�
-Encoder/encoder_mu/kernel/generator_opti/readIdentity(Encoder/encoder_mu/kernel/generator_opti*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	�d
�
LEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
�
BEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/ConstConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<Encoder/encoder_mu/kernel/generator_opti_1/Initializer/zerosFillLEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorBEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/Const*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*

index_type0*
_output_shapes
:	�d
�
*Encoder/encoder_mu/kernel/generator_opti_1
VariableV2*
shared_name *,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d
�
1Encoder/encoder_mu/kernel/generator_opti_1/AssignAssign*Encoder/encoder_mu/kernel/generator_opti_1<Encoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking(
�
/Encoder/encoder_mu/kernel/generator_opti_1/readIdentity*Encoder/encoder_mu/kernel/generator_opti_1*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	�d
�
8Encoder/encoder_mu/bias/generator_opti/Initializer/zerosConst**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0*
_output_shapes
:d
�
&Encoder/encoder_mu/bias/generator_opti
VariableV2*
shared_name **
_class 
loc:@Encoder/encoder_mu/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d
�
-Encoder/encoder_mu/bias/generator_opti/AssignAssign&Encoder/encoder_mu/bias/generator_opti8Encoder/encoder_mu/bias/generator_opti/Initializer/zeros*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0**
_class 
loc:@Encoder/encoder_mu/bias
�
+Encoder/encoder_mu/bias/generator_opti/readIdentity&Encoder/encoder_mu/bias/generator_opti*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d
�
:Encoder/encoder_mu/bias/generator_opti_1/Initializer/zerosConst**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0*
_output_shapes
:d
�
(Encoder/encoder_mu/bias/generator_opti_1
VariableV2*
dtype0*
_output_shapes
:d*
shared_name **
_class 
loc:@Encoder/encoder_mu/bias*
	container *
shape:d
�
/Encoder/encoder_mu/bias/generator_opti_1/AssignAssign(Encoder/encoder_mu/bias/generator_opti_1:Encoder/encoder_mu/bias/generator_opti_1/Initializer/zeros*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0**
_class 
loc:@Encoder/encoder_mu/bias
�
-Encoder/encoder_mu/bias/generator_opti_1/readIdentity(Encoder/encoder_mu/bias/generator_opti_1*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d
�
NEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
�
DEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/ConstConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
>Encoder/encoder_logvar/kernel/generator_opti/Initializer/zerosFillNEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/shape_as_tensorDEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/Const*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*

index_type0*
_output_shapes
:	�d
�
,Encoder/encoder_logvar/kernel/generator_opti
VariableV2*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	�d*
dtype0*
_output_shapes
:	�d
�
3Encoder/encoder_logvar/kernel/generator_opti/AssignAssign,Encoder/encoder_logvar/kernel/generator_opti>Encoder/encoder_logvar/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(*
_output_shapes
:	�d
�
1Encoder/encoder_logvar/kernel/generator_opti/readIdentity,Encoder/encoder_logvar/kernel/generator_opti*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	�d
�
PEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
�
FEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/ConstConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@Encoder/encoder_logvar/kernel/generator_opti_1/Initializer/zerosFillPEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorFEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/Const*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*

index_type0*
_output_shapes
:	�d
�
.Encoder/encoder_logvar/kernel/generator_opti_1
VariableV2*
dtype0*
_output_shapes
:	�d*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	�d
�
5Encoder/encoder_logvar/kernel/generator_opti_1/AssignAssign.Encoder/encoder_logvar/kernel/generator_opti_1@Encoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking(
�
3Encoder/encoder_logvar/kernel/generator_opti_1/readIdentity.Encoder/encoder_logvar/kernel/generator_opti_1*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	�d
�
<Encoder/encoder_logvar/bias/generator_opti/Initializer/zerosConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0*
_output_shapes
:d
�
*Encoder/encoder_logvar/bias/generator_opti
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape:d
�
1Encoder/encoder_logvar/bias/generator_opti/AssignAssign*Encoder/encoder_logvar/bias/generator_opti<Encoder/encoder_logvar/bias/generator_opti/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
:d
�
/Encoder/encoder_logvar/bias/generator_opti/readIdentity*Encoder/encoder_logvar/bias/generator_opti*
_output_shapes
:d*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias
�
>Encoder/encoder_logvar/bias/generator_opti_1/Initializer/zerosConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0*
_output_shapes
:d
�
,Encoder/encoder_logvar/bias/generator_opti_1
VariableV2*.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
�
3Encoder/encoder_logvar/bias/generator_opti_1/AssignAssign,Encoder/encoder_logvar/bias/generator_opti_1>Encoder/encoder_logvar/bias/generator_opti_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
:d
�
1Encoder/encoder_logvar/bias/generator_opti_1/readIdentity,Encoder/encoder_logvar/bias/generator_opti_1*
_output_shapes
:d*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias
�
[Decoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
�
QDecoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
KDecoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill[Decoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorQDecoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
9Decoder/first_layer/fully_connected/kernel/generator_opti
VariableV2*
dtype0*
_output_shapes
:	d�*
shared_name *=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
	container *
shape:	d�
�
@Decoder/first_layer/fully_connected/kernel/generator_opti/AssignAssign9Decoder/first_layer/fully_connected/kernel/generator_optiKDecoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
>Decoder/first_layer/fully_connected/kernel/generator_opti/readIdentity9Decoder/first_layer/fully_connected/kernel/generator_opti*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
]Decoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB"d      
�
SDecoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
MDecoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill]Decoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorSDecoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
;Decoder/first_layer/fully_connected/kernel/generator_opti_1
VariableV2*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name 
�
BDecoder/first_layer/fully_connected/kernel/generator_opti_1/AssignAssign;Decoder/first_layer/fully_connected/kernel/generator_opti_1MDecoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
@Decoder/first_layer/fully_connected/kernel/generator_opti_1/readIdentity;Decoder/first_layer/fully_connected/kernel/generator_opti_1*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
IDecoder/first_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
valueB�*    
�
7Decoder/first_layer/fully_connected/bias/generator_opti
VariableV2*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
>Decoder/first_layer/fully_connected/bias/generator_opti/AssignAssign7Decoder/first_layer/fully_connected/bias/generator_optiIDecoder/first_layer/fully_connected/bias/generator_opti/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias
�
<Decoder/first_layer/fully_connected/bias/generator_opti/readIdentity7Decoder/first_layer/fully_connected/bias/generator_opti*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
_output_shapes	
:�
�
KDecoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
9Decoder/first_layer/fully_connected/bias/generator_opti_1
VariableV2*
shared_name *;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
@Decoder/first_layer/fully_connected/bias/generator_opti_1/AssignAssign9Decoder/first_layer/fully_connected/bias/generator_opti_1KDecoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
>Decoder/first_layer/fully_connected/bias/generator_opti_1/readIdentity9Decoder/first_layer/fully_connected/bias/generator_opti_1*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
_output_shapes	
:�
�
\Decoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
RDecoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LDecoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill\Decoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorRDecoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*

index_type0
�
:Decoder/second_layer/fully_connected/kernel/generator_opti
VariableV2*
shared_name *>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
ADecoder/second_layer/fully_connected/kernel/generator_opti/AssignAssign:Decoder/second_layer/fully_connected/kernel/generator_optiLDecoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
?Decoder/second_layer/fully_connected/kernel/generator_opti/readIdentity:Decoder/second_layer/fully_connected/kernel/generator_opti*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
^Decoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
TDecoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NDecoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill^Decoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorTDecoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
<Decoder/second_layer/fully_connected/kernel/generator_opti_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
	container *
shape:
��
�
CDecoder/second_layer/fully_connected/kernel/generator_opti_1/AssignAssign<Decoder/second_layer/fully_connected/kernel/generator_opti_1NDecoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel
�
ADecoder/second_layer/fully_connected/kernel/generator_opti_1/readIdentity<Decoder/second_layer/fully_connected/kernel/generator_opti_1*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
JDecoder/second_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
valueB�*    
�
8Decoder/second_layer/fully_connected/bias/generator_opti
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
	container *
shape:�
�
?Decoder/second_layer/fully_connected/bias/generator_opti/AssignAssign8Decoder/second_layer/fully_connected/bias/generator_optiJDecoder/second_layer/fully_connected/bias/generator_opti/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
=Decoder/second_layer/fully_connected/bias/generator_opti/readIdentity8Decoder/second_layer/fully_connected/bias/generator_opti*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
_output_shapes	
:�
�
LDecoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
:Decoder/second_layer/fully_connected/bias/generator_opti_1
VariableV2*
shared_name *<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
ADecoder/second_layer/fully_connected/bias/generator_opti_1/AssignAssign:Decoder/second_layer/fully_connected/bias/generator_opti_1LDecoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
?Decoder/second_layer/fully_connected/bias/generator_opti_1/readIdentity:Decoder/second_layer/fully_connected/bias/generator_opti_1*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
_output_shapes	
:�
�
ODecoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zerosConst*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
=Decoder/second_layer/batch_normalization/gamma/generator_opti
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
	container *
shape:�
�
DDecoder/second_layer/batch_normalization/gamma/generator_opti/AssignAssign=Decoder/second_layer/batch_normalization/gamma/generator_optiODecoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
BDecoder/second_layer/batch_normalization/gamma/generator_opti/readIdentity=Decoder/second_layer/batch_normalization/gamma/generator_opti*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
QDecoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zerosConst*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
?Decoder/second_layer/batch_normalization/gamma/generator_opti_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma
�
FDecoder/second_layer/batch_normalization/gamma/generator_opti_1/AssignAssign?Decoder/second_layer/batch_normalization/gamma/generator_opti_1QDecoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
DDecoder/second_layer/batch_normalization/gamma/generator_opti_1/readIdentity?Decoder/second_layer/batch_normalization/gamma/generator_opti_1*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
NDecoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
valueB�*    
�
<Decoder/second_layer/batch_normalization/beta/generator_opti
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
	container *
shape:�
�
CDecoder/second_layer/batch_normalization/beta/generator_opti/AssignAssign<Decoder/second_layer/batch_normalization/beta/generator_optiNDecoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
ADecoder/second_layer/batch_normalization/beta/generator_opti/readIdentity<Decoder/second_layer/batch_normalization/beta/generator_opti*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
PDecoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zerosConst*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
>Decoder/second_layer/batch_normalization/beta/generator_opti_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
	container 
�
EDecoder/second_layer/batch_normalization/beta/generator_opti_1/AssignAssign>Decoder/second_layer/batch_normalization/beta/generator_opti_1PDecoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta
�
CDecoder/second_layer/batch_normalization/beta/generator_opti_1/readIdentity>Decoder/second_layer/batch_normalization/beta/generator_opti_1*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
EDecoder/dense/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@Decoder/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
;Decoder/dense/kernel/generator_opti/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *'
_class
loc:@Decoder/dense/kernel*
valueB
 *    
�
5Decoder/dense/kernel/generator_opti/Initializer/zerosFillEDecoder/dense/kernel/generator_opti/Initializer/zeros/shape_as_tensor;Decoder/dense/kernel/generator_opti/Initializer/zeros/Const*
T0*'
_class
loc:@Decoder/dense/kernel*

index_type0* 
_output_shapes
:
��
�
#Decoder/dense/kernel/generator_opti
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *'
_class
loc:@Decoder/dense/kernel
�
*Decoder/dense/kernel/generator_opti/AssignAssign#Decoder/dense/kernel/generator_opti5Decoder/dense/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Decoder/dense/kernel*
validate_shape(* 
_output_shapes
:
��
�
(Decoder/dense/kernel/generator_opti/readIdentity#Decoder/dense/kernel/generator_opti*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:
��
�
GDecoder/dense/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@Decoder/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
=Decoder/dense/kernel/generator_opti_1/Initializer/zeros/ConstConst*'
_class
loc:@Decoder/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7Decoder/dense/kernel/generator_opti_1/Initializer/zerosFillGDecoder/dense/kernel/generator_opti_1/Initializer/zeros/shape_as_tensor=Decoder/dense/kernel/generator_opti_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*'
_class
loc:@Decoder/dense/kernel*

index_type0
�
%Decoder/dense/kernel/generator_opti_1
VariableV2*
shared_name *'
_class
loc:@Decoder/dense/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
,Decoder/dense/kernel/generator_opti_1/AssignAssign%Decoder/dense/kernel/generator_opti_17Decoder/dense/kernel/generator_opti_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Decoder/dense/kernel*
validate_shape(* 
_output_shapes
:
��
�
*Decoder/dense/kernel/generator_opti_1/readIdentity%Decoder/dense/kernel/generator_opti_1*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:
��
�
3Decoder/dense/bias/generator_opti/Initializer/zerosConst*%
_class
loc:@Decoder/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
!Decoder/dense/bias/generator_opti
VariableV2*%
_class
loc:@Decoder/dense/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
(Decoder/dense/bias/generator_opti/AssignAssign!Decoder/dense/bias/generator_opti3Decoder/dense/bias/generator_opti/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes	
:�
�
&Decoder/dense/bias/generator_opti/readIdentity!Decoder/dense/bias/generator_opti*
_output_shapes	
:�*
T0*%
_class
loc:@Decoder/dense/bias
�
5Decoder/dense/bias/generator_opti_1/Initializer/zerosConst*%
_class
loc:@Decoder/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
#Decoder/dense/bias/generator_opti_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *%
_class
loc:@Decoder/dense/bias*
	container *
shape:�
�
*Decoder/dense/bias/generator_opti_1/AssignAssign#Decoder/dense/bias/generator_opti_15Decoder/dense/bias/generator_opti_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes	
:�
�
(Decoder/dense/bias/generator_opti_1/readIdentity#Decoder/dense/bias/generator_opti_1*
_output_shapes	
:�*
T0*%
_class
loc:@Decoder/dense/bias
a
generator_opti/learning_rateConst*
valueB
 *�Q9*
dtype0*
_output_shapes
: 
Y
generator_opti/beta1Const*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Y
generator_opti/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
[
generator_opti/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
Jgenerator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam*Encoder/first_layer/fully_connected/kernel9Encoder/first_layer/fully_connected/kernel/generator_opti;Encoder/first_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonVgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel
�
Hgenerator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam(Encoder/first_layer/fully_connected/bias7Encoder/first_layer/fully_connected/bias/generator_opti9Encoder/first_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
Kgenerator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Encoder/second_layer/fully_connected/kernel:Encoder/second_layer/fully_connected/kernel/generator_opti<Encoder/second_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
�
Igenerator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Encoder/second_layer/fully_connected/bias8Encoder/second_layer/fully_connected/bias/generator_opti:Encoder/second_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonXgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
Ngenerator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Encoder/second_layer/batch_normalization/gamma=Encoder/second_layer/batch_normalization/gamma/generator_opti?Encoder/second_layer/batch_normalization/gamma/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�
�
Mgenerator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Encoder/second_layer/batch_normalization/beta<Encoder/second_layer/batch_normalization/beta/generator_opti>Encoder/second_layer/batch_normalization/beta/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilon`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
9generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdam	ApplyAdamEncoder/encoder_mu/kernel(Encoder/encoder_mu/kernel/generator_opti*Encoder/encoder_mu/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonEgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
use_nesterov( *
_output_shapes
:	�d
�
7generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam	ApplyAdamEncoder/encoder_mu/bias&Encoder/encoder_mu/bias/generator_opti(Encoder/encoder_mu/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonFgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:d*
use_locking( *
T0**
_class 
loc:@Encoder/encoder_mu/bias
�
=generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam	ApplyAdamEncoder/encoder_logvar/kernel,Encoder/encoder_logvar/kernel/generator_opti.Encoder/encoder_logvar/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonIgradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
use_nesterov( *
_output_shapes
:	�d
�
;generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam	ApplyAdamEncoder/encoder_logvar/bias*Encoder/encoder_logvar/bias/generator_opti,Encoder/encoder_logvar/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonJgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
use_nesterov( *
_output_shapes
:d
�
Jgenerator_opti/update_Decoder/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam*Decoder/first_layer/fully_connected/kernel9Decoder/first_layer/fully_connected/kernel/generator_opti;Decoder/first_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonVgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	d�
�
Hgenerator_opti/update_Decoder/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam(Decoder/first_layer/fully_connected/bias7Decoder/first_layer/fully_connected/bias/generator_opti9Decoder/first_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
Kgenerator_opti/update_Decoder/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Decoder/second_layer/fully_connected/kernel:Decoder/second_layer/fully_connected/kernel/generator_opti<Decoder/second_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel
�
Igenerator_opti/update_Decoder/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Decoder/second_layer/fully_connected/bias8Decoder/second_layer/fully_connected/bias/generator_opti:Decoder/second_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonXgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
Ngenerator_opti/update_Decoder/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Decoder/second_layer/batch_normalization/gamma=Decoder/second_layer/batch_normalization/gamma/generator_opti?Decoder/second_layer/batch_normalization/gamma/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�
�
Mgenerator_opti/update_Decoder/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Decoder/second_layer/batch_normalization/beta<Decoder/second_layer/batch_normalization/beta/generator_opti>Decoder/second_layer/batch_normalization/beta/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilon`gradients_1/Decoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:�
�
4generator_opti/update_Decoder/dense/kernel/ApplyAdam	ApplyAdamDecoder/dense/kernel#Decoder/dense/kernel/generator_opti%Decoder/dense/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilon@gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency_1*
T0*'
_class
loc:@Decoder/dense/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
�
2generator_opti/update_Decoder/dense/bias/ApplyAdam	ApplyAdamDecoder/dense/bias!Decoder/dense/bias/generator_opti#Decoder/dense/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonAgradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*%
_class
loc:@Decoder/dense/bias
�
generator_opti/mulMulbeta1_power_1/readgenerator_opti/beta13^generator_opti/update_Decoder/dense/bias/ApplyAdam5^generator_opti/update_Decoder/dense/kernel/ApplyAdamI^generator_opti/update_Decoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Decoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Decoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Decoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Decoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Decoder/second_layer/fully_connected/kernel/ApplyAdam<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*%
_class
loc:@Decoder/dense/bias
�
generator_opti/AssignAssignbeta1_power_1generator_opti/mul*
use_locking( *
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes
: 
�
generator_opti/mul_1Mulbeta2_power_1/readgenerator_opti/beta23^generator_opti/update_Decoder/dense/bias/ApplyAdam5^generator_opti/update_Decoder/dense/kernel/ApplyAdamI^generator_opti/update_Decoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Decoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Decoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Decoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Decoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Decoder/second_layer/fully_connected/kernel/ApplyAdam<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam*
T0*%
_class
loc:@Decoder/dense/bias*
_output_shapes
: 
�
generator_opti/Assign_1Assignbeta2_power_1generator_opti/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*%
_class
loc:@Decoder/dense/bias
�

generator_optiNoOp^generator_opti/Assign^generator_opti/Assign_13^generator_opti/update_Decoder/dense/bias/ApplyAdam5^generator_opti/update_Decoder/dense/kernel/ApplyAdamI^generator_opti/update_Decoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Decoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Decoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Decoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Decoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Decoder/second_layer/fully_connected/kernel/ApplyAdam<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam
i
Merge/MergeSummaryMergeSummarygenerator_loss_1discriminator_loss*
N*
_output_shapes
: ""9
	summaries,
*
generator_loss_1:0
discriminator_loss:0"�&
trainable_variables�%�%
�
,Encoder/first_layer/fully_connected/kernel:01Encoder/first_layer/fully_connected/kernel/Assign1Encoder/first_layer/fully_connected/kernel/read:02GEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform:08
�
*Encoder/first_layer/fully_connected/bias:0/Encoder/first_layer/fully_connected/bias/Assign/Encoder/first_layer/fully_connected/bias/read:02<Encoder/first_layer/fully_connected/bias/Initializer/zeros:08
�
-Encoder/second_layer/fully_connected/kernel:02Encoder/second_layer/fully_connected/kernel/Assign2Encoder/second_layer/fully_connected/kernel/read:02HEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform:08
�
+Encoder/second_layer/fully_connected/bias:00Encoder/second_layer/fully_connected/bias/Assign0Encoder/second_layer/fully_connected/bias/read:02=Encoder/second_layer/fully_connected/bias/Initializer/zeros:08
�
0Encoder/second_layer/batch_normalization/gamma:05Encoder/second_layer/batch_normalization/gamma/Assign5Encoder/second_layer/batch_normalization/gamma/read:02AEncoder/second_layer/batch_normalization/gamma/Initializer/ones:08
�
/Encoder/second_layer/batch_normalization/beta:04Encoder/second_layer/batch_normalization/beta/Assign4Encoder/second_layer/batch_normalization/beta/read:02AEncoder/second_layer/batch_normalization/beta/Initializer/zeros:08
�
Encoder/encoder_mu/kernel:0 Encoder/encoder_mu/kernel/Assign Encoder/encoder_mu/kernel/read:026Encoder/encoder_mu/kernel/Initializer/random_uniform:08
�
Encoder/encoder_mu/bias:0Encoder/encoder_mu/bias/AssignEncoder/encoder_mu/bias/read:02+Encoder/encoder_mu/bias/Initializer/zeros:08
�
Encoder/encoder_logvar/kernel:0$Encoder/encoder_logvar/kernel/Assign$Encoder/encoder_logvar/kernel/read:02:Encoder/encoder_logvar/kernel/Initializer/random_uniform:08
�
Encoder/encoder_logvar/bias:0"Encoder/encoder_logvar/bias/Assign"Encoder/encoder_logvar/bias/read:02/Encoder/encoder_logvar/bias/Initializer/zeros:08
�
,Decoder/first_layer/fully_connected/kernel:01Decoder/first_layer/fully_connected/kernel/Assign1Decoder/first_layer/fully_connected/kernel/read:02GDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform:08
�
*Decoder/first_layer/fully_connected/bias:0/Decoder/first_layer/fully_connected/bias/Assign/Decoder/first_layer/fully_connected/bias/read:02<Decoder/first_layer/fully_connected/bias/Initializer/zeros:08
�
-Decoder/second_layer/fully_connected/kernel:02Decoder/second_layer/fully_connected/kernel/Assign2Decoder/second_layer/fully_connected/kernel/read:02HDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform:08
�
+Decoder/second_layer/fully_connected/bias:00Decoder/second_layer/fully_connected/bias/Assign0Decoder/second_layer/fully_connected/bias/read:02=Decoder/second_layer/fully_connected/bias/Initializer/zeros:08
�
0Decoder/second_layer/batch_normalization/gamma:05Decoder/second_layer/batch_normalization/gamma/Assign5Decoder/second_layer/batch_normalization/gamma/read:02ADecoder/second_layer/batch_normalization/gamma/Initializer/ones:08
�
/Decoder/second_layer/batch_normalization/beta:04Decoder/second_layer/batch_normalization/beta/Assign4Decoder/second_layer/batch_normalization/beta/read:02ADecoder/second_layer/batch_normalization/beta/Initializer/zeros:08
�
Decoder/dense/kernel:0Decoder/dense/kernel/AssignDecoder/dense/kernel/read:021Decoder/dense/kernel/Initializer/random_uniform:08
v
Decoder/dense/bias:0Decoder/dense/bias/AssignDecoder/dense/bias/read:02&Decoder/dense/bias/Initializer/zeros:08
�
2Discriminator/first_layer/fully_connected/kernel:07Discriminator/first_layer/fully_connected/kernel/Assign7Discriminator/first_layer/fully_connected/kernel/read:02MDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
�
0Discriminator/first_layer/fully_connected/bias:05Discriminator/first_layer/fully_connected/bias/Assign5Discriminator/first_layer/fully_connected/bias/read:02BDiscriminator/first_layer/fully_connected/bias/Initializer/zeros:08
�
3Discriminator/second_layer/fully_connected/kernel:08Discriminator/second_layer/fully_connected/kernel/Assign8Discriminator/second_layer/fully_connected/kernel/read:02NDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
�
1Discriminator/second_layer/fully_connected/bias:06Discriminator/second_layer/fully_connected/bias/Assign6Discriminator/second_layer/fully_connected/bias/read:02CDiscriminator/second_layer/fully_connected/bias/Initializer/zeros:08
�
Discriminator/prob/kernel:0 Discriminator/prob/kernel/Assign Discriminator/prob/kernel/read:026Discriminator/prob/kernel/Initializer/random_uniform:08
�
Discriminator/prob/bias:0Discriminator/prob/bias/AssignDiscriminator/prob/bias/read:02+Discriminator/prob/bias/Initializer/zeros:08"2
train_op&
$
discriminator_opti
generator_opti"��
	variables��
�
,Encoder/first_layer/fully_connected/kernel:01Encoder/first_layer/fully_connected/kernel/Assign1Encoder/first_layer/fully_connected/kernel/read:02GEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform:08
�
*Encoder/first_layer/fully_connected/bias:0/Encoder/first_layer/fully_connected/bias/Assign/Encoder/first_layer/fully_connected/bias/read:02<Encoder/first_layer/fully_connected/bias/Initializer/zeros:08
�
-Encoder/second_layer/fully_connected/kernel:02Encoder/second_layer/fully_connected/kernel/Assign2Encoder/second_layer/fully_connected/kernel/read:02HEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform:08
�
+Encoder/second_layer/fully_connected/bias:00Encoder/second_layer/fully_connected/bias/Assign0Encoder/second_layer/fully_connected/bias/read:02=Encoder/second_layer/fully_connected/bias/Initializer/zeros:08
�
0Encoder/second_layer/batch_normalization/gamma:05Encoder/second_layer/batch_normalization/gamma/Assign5Encoder/second_layer/batch_normalization/gamma/read:02AEncoder/second_layer/batch_normalization/gamma/Initializer/ones:08
�
/Encoder/second_layer/batch_normalization/beta:04Encoder/second_layer/batch_normalization/beta/Assign4Encoder/second_layer/batch_normalization/beta/read:02AEncoder/second_layer/batch_normalization/beta/Initializer/zeros:08
�
6Encoder/second_layer/batch_normalization/moving_mean:0;Encoder/second_layer/batch_normalization/moving_mean/Assign;Encoder/second_layer/batch_normalization/moving_mean/read:02HEncoder/second_layer/batch_normalization/moving_mean/Initializer/zeros:0
�
:Encoder/second_layer/batch_normalization/moving_variance:0?Encoder/second_layer/batch_normalization/moving_variance/Assign?Encoder/second_layer/batch_normalization/moving_variance/read:02KEncoder/second_layer/batch_normalization/moving_variance/Initializer/ones:0
�
Encoder/encoder_mu/kernel:0 Encoder/encoder_mu/kernel/Assign Encoder/encoder_mu/kernel/read:026Encoder/encoder_mu/kernel/Initializer/random_uniform:08
�
Encoder/encoder_mu/bias:0Encoder/encoder_mu/bias/AssignEncoder/encoder_mu/bias/read:02+Encoder/encoder_mu/bias/Initializer/zeros:08
�
Encoder/encoder_logvar/kernel:0$Encoder/encoder_logvar/kernel/Assign$Encoder/encoder_logvar/kernel/read:02:Encoder/encoder_logvar/kernel/Initializer/random_uniform:08
�
Encoder/encoder_logvar/bias:0"Encoder/encoder_logvar/bias/Assign"Encoder/encoder_logvar/bias/read:02/Encoder/encoder_logvar/bias/Initializer/zeros:08
�
,Decoder/first_layer/fully_connected/kernel:01Decoder/first_layer/fully_connected/kernel/Assign1Decoder/first_layer/fully_connected/kernel/read:02GDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform:08
�
*Decoder/first_layer/fully_connected/bias:0/Decoder/first_layer/fully_connected/bias/Assign/Decoder/first_layer/fully_connected/bias/read:02<Decoder/first_layer/fully_connected/bias/Initializer/zeros:08
�
-Decoder/second_layer/fully_connected/kernel:02Decoder/second_layer/fully_connected/kernel/Assign2Decoder/second_layer/fully_connected/kernel/read:02HDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform:08
�
+Decoder/second_layer/fully_connected/bias:00Decoder/second_layer/fully_connected/bias/Assign0Decoder/second_layer/fully_connected/bias/read:02=Decoder/second_layer/fully_connected/bias/Initializer/zeros:08
�
0Decoder/second_layer/batch_normalization/gamma:05Decoder/second_layer/batch_normalization/gamma/Assign5Decoder/second_layer/batch_normalization/gamma/read:02ADecoder/second_layer/batch_normalization/gamma/Initializer/ones:08
�
/Decoder/second_layer/batch_normalization/beta:04Decoder/second_layer/batch_normalization/beta/Assign4Decoder/second_layer/batch_normalization/beta/read:02ADecoder/second_layer/batch_normalization/beta/Initializer/zeros:08
�
6Decoder/second_layer/batch_normalization/moving_mean:0;Decoder/second_layer/batch_normalization/moving_mean/Assign;Decoder/second_layer/batch_normalization/moving_mean/read:02HDecoder/second_layer/batch_normalization/moving_mean/Initializer/zeros:0
�
:Decoder/second_layer/batch_normalization/moving_variance:0?Decoder/second_layer/batch_normalization/moving_variance/Assign?Decoder/second_layer/batch_normalization/moving_variance/read:02KDecoder/second_layer/batch_normalization/moving_variance/Initializer/ones:0
�
Decoder/dense/kernel:0Decoder/dense/kernel/AssignDecoder/dense/kernel/read:021Decoder/dense/kernel/Initializer/random_uniform:08
v
Decoder/dense/bias:0Decoder/dense/bias/AssignDecoder/dense/bias/read:02&Decoder/dense/bias/Initializer/zeros:08
�
2Discriminator/first_layer/fully_connected/kernel:07Discriminator/first_layer/fully_connected/kernel/Assign7Discriminator/first_layer/fully_connected/kernel/read:02MDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
�
0Discriminator/first_layer/fully_connected/bias:05Discriminator/first_layer/fully_connected/bias/Assign5Discriminator/first_layer/fully_connected/bias/read:02BDiscriminator/first_layer/fully_connected/bias/Initializer/zeros:08
�
3Discriminator/second_layer/fully_connected/kernel:08Discriminator/second_layer/fully_connected/kernel/Assign8Discriminator/second_layer/fully_connected/kernel/read:02NDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
�
1Discriminator/second_layer/fully_connected/bias:06Discriminator/second_layer/fully_connected/bias/Assign6Discriminator/second_layer/fully_connected/bias/read:02CDiscriminator/second_layer/fully_connected/bias/Initializer/zeros:08
�
Discriminator/prob/kernel:0 Discriminator/prob/kernel/Assign Discriminator/prob/kernel/read:026Discriminator/prob/kernel/Initializer/random_uniform:08
�
Discriminator/prob/bias:0Discriminator/prob/bias/AssignDiscriminator/prob/bias/read:02+Discriminator/prob/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
�
EDiscriminator/first_layer/fully_connected/kernel/discriminator_opti:0JDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/AssignJDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/read:02WDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros:0
�
GDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1:0LDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/AssignLDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/read:02YDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros:0
�
CDiscriminator/first_layer/fully_connected/bias/discriminator_opti:0HDiscriminator/first_layer/fully_connected/bias/discriminator_opti/AssignHDiscriminator/first_layer/fully_connected/bias/discriminator_opti/read:02UDiscriminator/first_layer/fully_connected/bias/discriminator_opti/Initializer/zeros:0
�
EDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1:0JDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/AssignJDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/read:02WDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros:0
�
FDiscriminator/second_layer/fully_connected/kernel/discriminator_opti:0KDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/AssignKDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/read:02XDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros:0
�
HDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1:0MDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/AssignMDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/read:02ZDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros:0
�
DDiscriminator/second_layer/fully_connected/bias/discriminator_opti:0IDiscriminator/second_layer/fully_connected/bias/discriminator_opti/AssignIDiscriminator/second_layer/fully_connected/bias/discriminator_opti/read:02VDiscriminator/second_layer/fully_connected/bias/discriminator_opti/Initializer/zeros:0
�
FDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1:0KDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/AssignKDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/read:02XDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros:0
�
.Discriminator/prob/kernel/discriminator_opti:03Discriminator/prob/kernel/discriminator_opti/Assign3Discriminator/prob/kernel/discriminator_opti/read:02@Discriminator/prob/kernel/discriminator_opti/Initializer/zeros:0
�
0Discriminator/prob/kernel/discriminator_opti_1:05Discriminator/prob/kernel/discriminator_opti_1/Assign5Discriminator/prob/kernel/discriminator_opti_1/read:02BDiscriminator/prob/kernel/discriminator_opti_1/Initializer/zeros:0
�
,Discriminator/prob/bias/discriminator_opti:01Discriminator/prob/bias/discriminator_opti/Assign1Discriminator/prob/bias/discriminator_opti/read:02>Discriminator/prob/bias/discriminator_opti/Initializer/zeros:0
�
.Discriminator/prob/bias/discriminator_opti_1:03Discriminator/prob/bias/discriminator_opti_1/Assign3Discriminator/prob/bias/discriminator_opti_1/read:02@Discriminator/prob/bias/discriminator_opti_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
�
;Encoder/first_layer/fully_connected/kernel/generator_opti:0@Encoder/first_layer/fully_connected/kernel/generator_opti/Assign@Encoder/first_layer/fully_connected/kernel/generator_opti/read:02MEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros:0
�
=Encoder/first_layer/fully_connected/kernel/generator_opti_1:0BEncoder/first_layer/fully_connected/kernel/generator_opti_1/AssignBEncoder/first_layer/fully_connected/kernel/generator_opti_1/read:02OEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros:0
�
9Encoder/first_layer/fully_connected/bias/generator_opti:0>Encoder/first_layer/fully_connected/bias/generator_opti/Assign>Encoder/first_layer/fully_connected/bias/generator_opti/read:02KEncoder/first_layer/fully_connected/bias/generator_opti/Initializer/zeros:0
�
;Encoder/first_layer/fully_connected/bias/generator_opti_1:0@Encoder/first_layer/fully_connected/bias/generator_opti_1/Assign@Encoder/first_layer/fully_connected/bias/generator_opti_1/read:02MEncoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zeros:0
�
<Encoder/second_layer/fully_connected/kernel/generator_opti:0AEncoder/second_layer/fully_connected/kernel/generator_opti/AssignAEncoder/second_layer/fully_connected/kernel/generator_opti/read:02NEncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros:0
�
>Encoder/second_layer/fully_connected/kernel/generator_opti_1:0CEncoder/second_layer/fully_connected/kernel/generator_opti_1/AssignCEncoder/second_layer/fully_connected/kernel/generator_opti_1/read:02PEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros:0
�
:Encoder/second_layer/fully_connected/bias/generator_opti:0?Encoder/second_layer/fully_connected/bias/generator_opti/Assign?Encoder/second_layer/fully_connected/bias/generator_opti/read:02LEncoder/second_layer/fully_connected/bias/generator_opti/Initializer/zeros:0
�
<Encoder/second_layer/fully_connected/bias/generator_opti_1:0AEncoder/second_layer/fully_connected/bias/generator_opti_1/AssignAEncoder/second_layer/fully_connected/bias/generator_opti_1/read:02NEncoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zeros:0
�
?Encoder/second_layer/batch_normalization/gamma/generator_opti:0DEncoder/second_layer/batch_normalization/gamma/generator_opti/AssignDEncoder/second_layer/batch_normalization/gamma/generator_opti/read:02QEncoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zeros:0
�
AEncoder/second_layer/batch_normalization/gamma/generator_opti_1:0FEncoder/second_layer/batch_normalization/gamma/generator_opti_1/AssignFEncoder/second_layer/batch_normalization/gamma/generator_opti_1/read:02SEncoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zeros:0
�
>Encoder/second_layer/batch_normalization/beta/generator_opti:0CEncoder/second_layer/batch_normalization/beta/generator_opti/AssignCEncoder/second_layer/batch_normalization/beta/generator_opti/read:02PEncoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zeros:0
�
@Encoder/second_layer/batch_normalization/beta/generator_opti_1:0EEncoder/second_layer/batch_normalization/beta/generator_opti_1/AssignEEncoder/second_layer/batch_normalization/beta/generator_opti_1/read:02REncoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zeros:0
�
*Encoder/encoder_mu/kernel/generator_opti:0/Encoder/encoder_mu/kernel/generator_opti/Assign/Encoder/encoder_mu/kernel/generator_opti/read:02<Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros:0
�
,Encoder/encoder_mu/kernel/generator_opti_1:01Encoder/encoder_mu/kernel/generator_opti_1/Assign1Encoder/encoder_mu/kernel/generator_opti_1/read:02>Encoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros:0
�
(Encoder/encoder_mu/bias/generator_opti:0-Encoder/encoder_mu/bias/generator_opti/Assign-Encoder/encoder_mu/bias/generator_opti/read:02:Encoder/encoder_mu/bias/generator_opti/Initializer/zeros:0
�
*Encoder/encoder_mu/bias/generator_opti_1:0/Encoder/encoder_mu/bias/generator_opti_1/Assign/Encoder/encoder_mu/bias/generator_opti_1/read:02<Encoder/encoder_mu/bias/generator_opti_1/Initializer/zeros:0
�
.Encoder/encoder_logvar/kernel/generator_opti:03Encoder/encoder_logvar/kernel/generator_opti/Assign3Encoder/encoder_logvar/kernel/generator_opti/read:02@Encoder/encoder_logvar/kernel/generator_opti/Initializer/zeros:0
�
0Encoder/encoder_logvar/kernel/generator_opti_1:05Encoder/encoder_logvar/kernel/generator_opti_1/Assign5Encoder/encoder_logvar/kernel/generator_opti_1/read:02BEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros:0
�
,Encoder/encoder_logvar/bias/generator_opti:01Encoder/encoder_logvar/bias/generator_opti/Assign1Encoder/encoder_logvar/bias/generator_opti/read:02>Encoder/encoder_logvar/bias/generator_opti/Initializer/zeros:0
�
.Encoder/encoder_logvar/bias/generator_opti_1:03Encoder/encoder_logvar/bias/generator_opti_1/Assign3Encoder/encoder_logvar/bias/generator_opti_1/read:02@Encoder/encoder_logvar/bias/generator_opti_1/Initializer/zeros:0
�
;Decoder/first_layer/fully_connected/kernel/generator_opti:0@Decoder/first_layer/fully_connected/kernel/generator_opti/Assign@Decoder/first_layer/fully_connected/kernel/generator_opti/read:02MDecoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros:0
�
=Decoder/first_layer/fully_connected/kernel/generator_opti_1:0BDecoder/first_layer/fully_connected/kernel/generator_opti_1/AssignBDecoder/first_layer/fully_connected/kernel/generator_opti_1/read:02ODecoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros:0
�
9Decoder/first_layer/fully_connected/bias/generator_opti:0>Decoder/first_layer/fully_connected/bias/generator_opti/Assign>Decoder/first_layer/fully_connected/bias/generator_opti/read:02KDecoder/first_layer/fully_connected/bias/generator_opti/Initializer/zeros:0
�
;Decoder/first_layer/fully_connected/bias/generator_opti_1:0@Decoder/first_layer/fully_connected/bias/generator_opti_1/Assign@Decoder/first_layer/fully_connected/bias/generator_opti_1/read:02MDecoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zeros:0
�
<Decoder/second_layer/fully_connected/kernel/generator_opti:0ADecoder/second_layer/fully_connected/kernel/generator_opti/AssignADecoder/second_layer/fully_connected/kernel/generator_opti/read:02NDecoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros:0
�
>Decoder/second_layer/fully_connected/kernel/generator_opti_1:0CDecoder/second_layer/fully_connected/kernel/generator_opti_1/AssignCDecoder/second_layer/fully_connected/kernel/generator_opti_1/read:02PDecoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros:0
�
:Decoder/second_layer/fully_connected/bias/generator_opti:0?Decoder/second_layer/fully_connected/bias/generator_opti/Assign?Decoder/second_layer/fully_connected/bias/generator_opti/read:02LDecoder/second_layer/fully_connected/bias/generator_opti/Initializer/zeros:0
�
<Decoder/second_layer/fully_connected/bias/generator_opti_1:0ADecoder/second_layer/fully_connected/bias/generator_opti_1/AssignADecoder/second_layer/fully_connected/bias/generator_opti_1/read:02NDecoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zeros:0
�
?Decoder/second_layer/batch_normalization/gamma/generator_opti:0DDecoder/second_layer/batch_normalization/gamma/generator_opti/AssignDDecoder/second_layer/batch_normalization/gamma/generator_opti/read:02QDecoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zeros:0
�
ADecoder/second_layer/batch_normalization/gamma/generator_opti_1:0FDecoder/second_layer/batch_normalization/gamma/generator_opti_1/AssignFDecoder/second_layer/batch_normalization/gamma/generator_opti_1/read:02SDecoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zeros:0
�
>Decoder/second_layer/batch_normalization/beta/generator_opti:0CDecoder/second_layer/batch_normalization/beta/generator_opti/AssignCDecoder/second_layer/batch_normalization/beta/generator_opti/read:02PDecoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zeros:0
�
@Decoder/second_layer/batch_normalization/beta/generator_opti_1:0EDecoder/second_layer/batch_normalization/beta/generator_opti_1/AssignEDecoder/second_layer/batch_normalization/beta/generator_opti_1/read:02RDecoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zeros:0
�
%Decoder/dense/kernel/generator_opti:0*Decoder/dense/kernel/generator_opti/Assign*Decoder/dense/kernel/generator_opti/read:027Decoder/dense/kernel/generator_opti/Initializer/zeros:0
�
'Decoder/dense/kernel/generator_opti_1:0,Decoder/dense/kernel/generator_opti_1/Assign,Decoder/dense/kernel/generator_opti_1/read:029Decoder/dense/kernel/generator_opti_1/Initializer/zeros:0
�
#Decoder/dense/bias/generator_opti:0(Decoder/dense/bias/generator_opti/Assign(Decoder/dense/bias/generator_opti/read:025Decoder/dense/bias/generator_opti/Initializer/zeros:0
�
%Decoder/dense/bias/generator_opti_1:0*Decoder/dense/bias/generator_opti_1/Assign*Decoder/dense/bias/generator_opti_1/read:027Decoder/dense/bias/generator_opti_1/Initializer/zeros:0��+�       �{�	z�����A*�
u
generator_loss_1*a	   ��j�?   ��j�?      �?!   ��j�?) G%���?2�iZ�?+�;$�?�������:              �?        
w
discriminator_loss*a	    Tt�?    Tt�?      �?!    Tt�?)  ��&�?2�P�1���?3?��|�?�������:              �?        ����       ۞��	t����A(*�
u
generator_loss_1*a	   �z�?   �z�?      �?!   �z�?) ���#�?2�Z�_���?����?�������:              �?        
w
discriminator_loss*a	    ���?    ���?      �?!    ���?)@8�}k�W?2�uS��a�?`��a�8�?�������:              �?        �S6a�       ۞��	%���AP*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) au���?2����?_&A�o��?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) ���&?2#�+(�ŉ?�7c_XY�?�������:              �?        �"ڭ�       ۞��	|�f���Ax*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) !.)�M�?2�Z�_���?����?�������:              �?        
w
discriminator_loss*a	   @d|?   @d|?      �?!   @d|?)�x�_0	?2o��5sz?���T}?�������:              �?        z����       S�	������A�*�
u
generator_loss_1*a	   �ӱ�?   �ӱ�?      �?!   �ӱ�?)@��*`k�?2�Z�_���?����?�������:              �?        
w
discriminator_loss*a	   �O�}?   �O�}?      �?!   �O�}?) `�vd�?2���T}?>	� �?�������:              �?        .�       S�	^���A�*�
u
generator_loss_1*a	   �8��?   �8��?      �?!   �8��?)@f��f>�?2�Z�_���?����?�������:              �?        
w
discriminator_loss*a	    �}?    �}?      �?!    �}?) �9��S
?2o��5sz?���T}?�������:              �?        ��7:�       S�	�;���A�*�
u
generator_loss_1*a	   �u��?   �u��?      �?!   �u��?) ]�D5E�?2�K?�?�Z�_���?�������:              �?        
w
discriminator_loss*a	   ���r?   ���r?      �?!   ���r?) d�L���>2uWy��r?hyO�s?�������:              �?        "����       S�	P`����A�*�
u
generator_loss_1*a	   �~��?   �~��?      �?!   �~��?) �U�b�?2�K?�?�Z�_���?�������:              �?        
w
discriminator_loss*a	   ���v?   ���v?      �?!   ���v?) "甪c ?2&b՞
�u?*QH�x?�������:              �?        ���c�       S�	�7���A�*�
u
generator_loss_1*a	    a%�?    a%�?      �?!    a%�?) vY?�?2�@�"��?�K?�?�������:              �?        
w
discriminator_loss*a	   ��~?   ��~?      �?!   ��~?) �p��9?2���T}?>	� �?�������:              �?        �<�#�       S�	Jl9���A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?)��l�q�?2�@�"��?�K?�?�������:              �?        
w
discriminator_loss*a	    [�}?    [�}?      �?!    [�}?) �BC�?2���T}?>	� �?�������:              �?        rJ?��       S�	$l����A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?)��3����?2�QK|:�?�@�"��?�������:              �?        
w
discriminator_loss*a	    H��?    H��?      �?!    H��?)  D^�u?2����=��?���J�\�?�������:              �?        �W}��       S�	'����A�*�
u
generator_loss_1*a	   @b��?   @b��?      �?!   @b��?)�����?2�QK|:�?�@�"��?�������:              �?        
w
discriminator_loss*a	   `ڸ�?   `ڸ�?      �?!   `ڸ�?) ="esc?2�g���w�?���g��?�������:              �?        ��i�       S�	F�Q���A�*�
u
generator_loss_1*a	   `h��?   `h��?      �?!   `h��?)@�Cn�?2�?>8s2�?yD$��?�������:              �?        
w
discriminator_loss*a	    >Dp?    >Dp?      �?!    >Dp?) @����>2�N�W�m?;8�clp?�������:              �?        �9���       S�	�Z����A�*�
u
generator_loss_1*a	   �&q�?   �&q�?      �?!   �&q�?)@>�iJz�?2yD$��?�QK|:�?�������:              �?        
w
discriminator_loss*a	   @E]�?   @E]�?      �?!   @E]�?)��c��e?2�g���w�?���g��?�������:              �?        #�3��       S�	�#���A�*�
u
generator_loss_1*a	   @a*�?   @a*�?      �?!   @a*�?) ߈���?2yD$��?�QK|:�?�������:              �?        
w
discriminator_loss*a	   �p�y?   �p�y?      �?!   �p�y?)�D!���?2*QH�x?o��5sz?�������:              �?        E����       S�	�t����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) �?�%�?2�?>8s2�?yD$��?�������:              �?        
w
discriminator_loss*a	    �+�?    �+�?      �?!    �+�?)@DFhInY?2�uS��a�?`��a�8�?�������:              �?        GɨL�       S�	�����A�*�
u
generator_loss_1*a	   �,��?   �,��?      �?!   �,��?) ě<��?2yD$��?�QK|:�?�������:              �?        
w
discriminator_loss*a	   �$�v?   �$�v?      �?!   �$�v?) ~"�� ?2&b՞
�u?*QH�x?�������:              �?        Ѓ���       S�	�w����A�*�
u
generator_loss_1*a	   �"��?   �"��?      �?!   �"��?)@�>Z۩�?2yD$��?�QK|:�?�������:              �?        
w
discriminator_loss*a	   �"�z?   �"�z?      �?!   �"�z?)��qL�?2o��5sz?���T}?�������:              �?        ��h+�       S�	?����A�*�
u
generator_loss_1*a	   ��U�?   ��U�?      �?!   ��U�?) !��]�?2�?>8s2�?yD$��?�������:              �?        
w
discriminator_loss*a	   `)�?   `)�?      �?!   `)�?) MRZI?2���T}?>	� �?�������:              �?        �����       S�	OM����A�*�
u
generator_loss_1*a	    t��?    t��?      �?!    t��?)@�:����?2Ӗ8��s�?�?>8s2�?�������:              �?        
w
discriminator_loss*a	   `�
�?   `�
�?      �?!   `�
�?)@V�6��v?2��]$A�?�{ �ǳ�?�������:              �?        �\j�       S�	�����A�*�
u
generator_loss_1*a	   @�4�?   @�4�?      �?!   @�4�?) I�}��?2�?>8s2�?yD$��?�������:              �?        
w
discriminator_loss*a	   �V�w?   �V�w?      �?!   �V�w?) ��r^?2&b՞
�u?*QH�x?�������:              �?        �?�       S�	w3����A�*�
u
generator_loss_1*a	   `���?   `���?      �?!   `���?)@�k&?��?2Ӗ8��s�?�?>8s2�?�������:              �?        
w
discriminator_loss*a	   @ېP?   @ېP?      �?!   @ېP?) i<��&�>2k�1^�sO?nK���LQ?�������:              �?        �g��       �{�	.���A*�
u
generator_loss_1*a	   �l��?   �l��?      �?!   �l��?)@ܠGe��?2Ӗ8��s�?�?>8s2�?�������:              �?        
w
discriminator_loss*a	   @gJ�?   @gJ�?      �?!   @gJ�?) I��T?2���J�\�?-Ա�L�?�������:              �?        �u���       ۞��	�G����A(*�
u
generator_loss_1*a	   ��s�?   ��s�?      �?!   ��s�?)@z��xG�?2Ӗ8��s�?�?>8s2�?�������:              �?        
w
discriminator_loss*a	   `��p?   `��p?      �?!   `��p?)@B�w��>2;8�clp?uWy��r?�������:              �?        P2q��       ۞��	�S���AP*�
u
generator_loss_1*a	   @2��?   @2��?      �?!   @2��?) ѥ��?2Ӗ8��s�?�?>8s2�?�������:              �?        
w
discriminator_loss*a	   ���h?   ���h?      �?!   ���h?) x�Y���>2Tw��Nof?P}���h?�������:              �?        �9���       ۞��	>���Ax*�
u
generator_loss_1*a	   �f��?   �f��?      �?!   �f��?) ٳ��̓?2Ӗ8��s�?�?>8s2�?�������:              �?        
w
discriminator_loss*a	   ��Q?   ��Q?      �?!   ��Q?)@���/�>2k�1^�sO?nK���LQ?�������:              �?        ��ϊ�       S�	�[����A�*�
u
generator_loss_1*a	    �&�?    �&�?      �?!    �&�?)@�+zb�?2!�����?Ӗ8��s�?�������:              �?        
w
discriminator_loss*a	   �v�?   �v�?      �?!   �v�?)@4�8mL6?2�Rc�ݒ?^�S���?�������:              �?        *� �       S�	''!���A�*�
u
generator_loss_1*a	    lp�?    lp�?      �?!    lp�?) �l5\�?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	   ��(`?   ��(`?      �?!   ��(`?)@��R�>2E��{��^?�l�P�`?�������:              �?        ���       S�	ݝ����A�*�
u
generator_loss_1*a	   �A@�?   �A@�?      �?!   �A@�?)��k ��?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	    b>i?    b>i?      �?!    b>i?)  ����>2P}���h?ߤ�(g%k?�������:              �?        v��L�       S�	��a���A�*�
u
generator_loss_1*a	    Ax�?    Ax�?      �?!    Ax�?)@�
��?2!�����?Ӗ8��s�?�������:              �?        
w
discriminator_loss*a	   `,b?   `,b?      �?!   `,b?)@J/ӫ��>2�l�P�`?���%��b?�������:              �?        ����       S�	J����A�*�
u
generator_loss_1*a	   @�{�?   @�{�?      �?!   @�{�?) 	f�;��?2!�����?Ӗ8��s�?�������:              �?        
w
discriminator_loss*a	   `�N^?   `�N^?      �?!   `�N^?) }�$_��>2�m9�H�[?E��{��^?�������:              �?        ����       S�	�- ��A�*�
u
generator_loss_1*a	   �I��?   �I��?      �?!   �I��?) Ұ`F�?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	   @�f?   @�f?      �?!   @�f?) I��D�>25Ucv0ed?Tw��Nof?�������:              �?        ����       S�	��r���A�*�
u
generator_loss_1*a	    �,�?    �,�?      �?!    �,�?)@���Y�?2!�����?Ӗ8��s�?�������:              �?        
w
discriminator_loss*a	   �]?   �]?      �?!   �]?) �~ʑ��>2�m9�H�[?E��{��^?�������:              �?        �k���       S�	��&���A�*�
u
generator_loss_1*a	   �3/�?   �3/�?      �?!   �3/�?) a���^�?2!�����?Ӗ8��s�?�������:              �?        
w
discriminator_loss*a	   �T�J?   �T�J?      �?!   �T�J?) ���	�>2�qU���I?IcD���L?�������:              �?        �����       S�	Cjޢ��A�*�
u
generator_loss_1*a	   �Y��?   �Y��?      �?!   �Y��?) E;Os�?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	   ��qd?   ��qd?      �?!   ��qd?)@��c�>25Ucv0ed?Tw��Nof?�������:              �?        _�\�       S�	�2����A�*�
u
generator_loss_1*a	   ��9�?   ��9�?      �?!   ��9�?)@$W�8��?2!�����?Ӗ8��s�?�������:              �?        
w
discriminator_loss*a	   ��b?   ��b?      �?!   ��b?) D��R|�>2�l�P�`?���%��b?�������:              �?        ��[�       S�	��L���A�*�
u
generator_loss_1*a	   ��Z�?   ��Z�?      �?!   ��Z�?) �qq��?2!�����?Ӗ8��s�?�������:              �?        
w
discriminator_loss*a	    ��N?    ��N?      �?!    ��N?)  �����>2IcD���L?k�1^�sO?�������:              �?        �Z��       S�	S�	���A�*�
u
generator_loss_1*a	    @��?    @��?      �?!    @��?)  �l���?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	    %�]?    %�]?      �?!    %�]?) �����>2�m9�H�[?E��{��^?�������:              �?        g��       S�	dΥ��A�*�
u
generator_loss_1*a	   �?   �?      �?!   �?) a��.�?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	   ��iF?   ��iF?      �?!   ��iF?) !��e�>2a�$��{E?
����G?�������:              �?        �T0�       S�	/����A�*�
u
generator_loss_1*a	    "H�?    "H�?      �?!    "H�?) d��⧌?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	   ��SL?   ��SL?      �?!   ��SL?)��Tܟ�>2�qU���I?IcD���L?�������:              �?        ����       S�	M�U���A�*�
u
generator_loss_1*a	    �'�?    �'�?      �?!    �'�?) Mv�ň?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	    |UQ?    |UQ?      �?!    |UQ?)  A9pǲ>2nK���LQ?�lDZrS?�������:              �?        X-��       S�	�����A�*�
u
generator_loss_1*a	   �<?�?   �<?�?      �?!   �<?�?) b�f��?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	   �L�E?   �L�E?      �?!   �L�E?) ) X|��>2a�$��{E?
����G?�������:              �?        qFO��       S�	(����A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?) h�+ �?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	   @�F?   @�F?      �?!   @�F?)���Y_�>2a�$��{E?
����G?�������:              �?        :�kF�       S�	TƩ��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) �|�;�?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	    S?    S?      �?!    S?)@ޭe��>2�lDZrS?<DKc��T?�������:              �?        �R���       �{�	�\����A*�
u
generator_loss_1*a	    3��?    3��?      �?!    3��?) H���҇?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   �\�A?   �\�A?      �?!   �\�A?) Ć��h�>2���#@?�!�A?�������:              �?        3���       ۞��	�h���A(*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) q/��?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   ��v;?   ��v;?      �?!   ��v;?) 2��W��>2��%>��:?d�\D�X=?�������:              �?        ��8��       ۞��	��E���AP*�
u
generator_loss_1*a	   �n�?   �n�?      �?!   �n�?)��_��?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	   ��1K?   ��1K?      �?!   ��1K?) ��k��>2�qU���I?IcD���L?�������:              �?        �j �       ۞��	�'���Ax*�
u
generator_loss_1*a	    �?    �?      �?!    �?) ND1K	�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   �j�E?   �j�E?      �?!   �j�E?) ��ٔ�>2a�$��{E?
����G?�������:              �?        �8�r�       S�	~y���A�*�
u
generator_loss_1*a	   ��`�?   ��`�?      �?!   ��`�?) R�V���?2��(!�ؼ?!�����?�������:              �?        
w
discriminator_loss*a	   �N~^?   �N~^?      �?!   �N~^?) j�q��>2�m9�H�[?E��{��^?�������:              �?        Y�o	�       S�	�Q����A�*�
u
generator_loss_1*a	   ��d�?   ��d�?      �?!   ��d�?) h�8�ą?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   �E?   �E?      �?!   �E?)@��>2�T���C?a�$��{E?�������:              �?        �\���       S�	������A�*�
u
generator_loss_1*a	   �)��?   �)��?      �?!   �)��?) ̞6�w�?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   `K?   `K?      �?!   `K?) ��P�>2�qU���I?IcD���L?�������:              �?        ��^��       S�	�Y߰��A�*�
u
generator_loss_1*a	   ��<�?   ��<�?      �?!   ��<�?)��t[��?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   �~�B?   �~�B?      �?!   �~�B?) $�Uw�>2�!�A?�T���C?�������:              �?        �8���       S�	3Pӱ��A�*�
u
generator_loss_1*a	   @ߩ�?   @ߩ�?      �?!   @ߩ�?)��)��7�?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	    �X3?    �X3?      �?!    �X3?) ���dw>2��82?�u�w74?�������:              �?        �=`��       S�	A˲��A�*�
u
generator_loss_1*a	    5û?    5û?      �?!    5û?) �G>�?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   �UiP?   �UiP?      �?!   �UiP?)@:��`հ>2k�1^�sO?nK���LQ?�������:              �?        A�J�       S�	�mȳ��A�*�
u
generator_loss_1*a	   ��˻?   ��˻?      �?!   ��˻?)����$�?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   @+�7?   @+�7?      �?!   @+�7?)�t ���>2��%�V6?uܬ�@8?�������:              �?        +�S��       S�	��Ǵ��A�*�
u
generator_loss_1*a	   �b��?   �b��?      �?!   �b��?) �ßd��?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   ��sR?   ��sR?      �?!   ��sR?) !a2?H�>2nK���LQ?�lDZrS?�������:              �?        7�+��       S�	�ɵ��A�*�
u
generator_loss_1*a	    �ӻ?    �ӻ?      �?!    �ӻ?) ��0�2�?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	    ��D?    ��D?      �?!    ��D?) ���zC�>2�T���C?a�$��{E?�������:              �?        L� ��       S�	��ζ��A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?) �<!��?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   � �M?   � �M?      �?!   � �M?)�x���>2IcD���L?k�1^�sO?�������:              �?        F:���       S�	8ڷ��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) H}��Q�?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	    |G?    |G?      �?!    |G?) x*E��>2a�$��{E?
����G?�������:              �?        ���I�       S�	_p���A�*�
u
generator_loss_1*a	    b|�?    b|�?      �?!    b|�?) ��<L�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   ��1?   ��1?      �?!   ��1?) �^ojs>2��bȬ�0?��82?�������:              �?        �>��       S�	�9����A�*�
u
generator_loss_1*a	   �gz�?   �gz�?      �?!   �gz�?) 0]	p��?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   �5�G?   �5�G?      �?!   �5�G?)�H6i�r�>2a�$��{E?
����G?�������:              �?        p� �       S�	�$���A�*�
u
generator_loss_1*a	   �3��?   �3��?      �?!   �3��?) �A��?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	    ��E?    ��E?      �?!    ��E?)@�6K��>2a�$��{E?
����G?�������:              �?        oΌ��       S�	_�/���A�*�
u
generator_loss_1*a	    [-�?    [-�?      �?!    [-�?) ��ψ?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   ���2?   ���2?      �?!   ���2?)@8U���u>2��82?�u�w74?�������:              �?        
�       S�	x0K���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)����-��?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   `-Q%?   `-Q%?      �?!   `-Q%?)@����f\>2U�4@@�$?+A�F�&?�������:              �?        Kbu�       S�	�q���A�*�
u
generator_loss_1*a	   �G��?   �G��?      �?!   �G��?) p�^z�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	     �$?     �$?      �?!     �$?)@�����Z>2U�4@@�$?+A�F�&?�������:              �?        xO�,�       S�	N����A�*�
u
generator_loss_1*a	    _߹?    _߹?      �?!    _߹?) *��?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   `�$?   `�$?      �?!   `�$?)@[��=Y>2�[^:��"?U�4@@�$?�������:              �?        ���I�       �{�	b�����A*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?)�`5ŘI�?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   �s"?   �s"?      �?!   �s"?)@���vGU>2�S�F !?�[^:��"?�������:              �?        �#�D�       ۞��	n����A(*�
u
generator_loss_1*a	   �oI�?   �oI�?      �?!   �oI�?)�@b'�n�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   @�b)?   @�b)?      �?!   @�b)?)�l�r#d>2I�I�)�(?�7Kaa+?�������:              �?        � ���       ۞��	MN#���AP*�
u
generator_loss_1*a	   ��-�?   ��-�?      �?!   ��-�?) �3�Ј?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   `��>?   `��>?      �?!   `��>?) �TU��>2d�\D�X=?���#@?�������:              �?        �?�8�       ۞��	1\���Ax*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) B�}fc�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   �I '?   �I '?      �?!   �I '?) ��2j�`>2+A�F�&?I�I�)�(?�������:              �?        ���a�       S�	Tn����A�*�
u
generator_loss_1*a	   �`*�?   �`*�?      �?!   �`*�?) s�nʃ?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	    5$!?    5$!?      �?!    5$!?) �/�B]R>2�S�F !?�[^:��"?�������:              �?        ���/�       S�	�G����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) �jc݁�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	    k�L?    k�L?      �?!    k�L?) �u�61�>2IcD���L?k�1^�sO?�������:              �?        cEq0�       S�	Ug5���A�*�
u
generator_loss_1*a	   ��͸?   ��͸?      �?!   ��͸?)�L�*�9�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	    �?    �?      �?!    �?) H�C=J>2�.�?ji6�9�?�������:              �?        �����       S�	&�����A�*�
u
generator_loss_1*a	   @�>�?   @�>�?      �?!   @�>�?)��Zja��?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)�(���M>2�.�?ji6�9�?�������:              �?        .`��       S�	������A�*�
u
generator_loss_1*a	   �?��?   �?��?      �?!   �?��?)� R�"�?2%g�cE9�?��(!�ؼ?�������:              �?        
w
discriminator_loss*a	   `��?   `��?      �?!   `��?) ��i\C>2�vV�R9?��ڋ?�������:              �?        r����       S�	�l���A�*�
u
generator_loss_1*a	   �t��?   �t��?      �?!   �t��?) �[ʂ?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   �}�?   �}�?      �?!   �}�?)@He�?>2�T7��?�vV�R9?�������:              �?        (
��       S�	I����A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �s��?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   `�R+?   `�R+?      �?!   `�R+?) �u�Tg>2I�I�)�(?�7Kaa+?�������:              �?        �
��       S�	s!���A�*�
u
generator_loss_1*a	   @j"�?   @j"�?      �?!   @j"�?)��(�X�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@aN�8>2�5�i}1?�T7��?�������:              �?        c2;��       S�	[6����A�*�
u
generator_loss_1*a	   @�̸?   @�̸?      �?!   @�̸?)��/$8�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   `?   `?      �?!   `?)@f9>2�5�i}1?�T7��?�������:              �?        ���y�       S�	�L����A�*�
u
generator_loss_1*a	   �oY�?   �oY�?      �?!   �oY�?) aol�	�?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �r�/?   �r�/?      �?!   �r�/?) b���o>2��VlQ.?��bȬ�0?�������:              �?        I�r��       S�	�j|���A�*�
u
generator_loss_1*a	   �ʸ?   �ʸ?      �?!   �ʸ?) �5�5�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	    \�%?    \�%?      �?!    \�%?)  �dUC]>2U�4@@�$?+A�F�&?�������:              �?        ��t��       S�	�n����A�*�
u
generator_loss_1*a	   �!�?   �!�?      �?!   �!�?) �A�ʻ�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	    �;E?    �;E?      �?!    �;E?)@���.�>2�T���C?a�$��{E?�������:              �?        Y�w��       S�	 bY���A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) <#�}��?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �A�;?   �A�;?      �?!   �A�;?) ���ˇ>2��%>��:?d�\D�X=?�������:              �?        ^dqo�       S�	�����A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?) &����?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	    �.?    �.?      �?!    �.?)  ��xL>2�.�?ji6�9�?�������:              �?        `�       S�	@�7���A�*�
u
generator_loss_1*a	    ]�?    ]�?      �?!    ]�?)@^�m~?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �.�-?   �.�-?      �?!   �.�-?)�L�d��k>2�7Kaa+?��VlQ.?�������:              �?        �����       S�	y�����A�*�
u
generator_loss_1*a	   �Mb�?   �Mb�?      �?!   �Mb�?)���x�"�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   ��#1?   ��#1?      �?!   ��#1?) ���[r>2��bȬ�0?��82?�������:              �?        �@�l�       S�	7����A�*�
u
generator_loss_1*a	   ��s�?   ��s�?      �?!   ��s�?) )�Z_�?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   @?   @?      �?!   @?)�|��+N>2ji6�9�?�S�F !?�������:              �?        ?���       S�	l����A�*�
u
generator_loss_1*a	    L��?    L��?      �?!    L��?) 3-�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   @�!?   @�!?      �?!   @�!?) Y���>>2�T7��?�vV�R9?�������:              �?        ��n��       �{�	�����A*�
u
generator_loss_1*a	    $�?    $�?      �?!    $�?) �(�B�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   �E�2?   �E�2?      �?!   �E�2?)@��H%wu>2��82?�u�w74?�������:              �?        c����       ۞��	��h���A(*�
u
generator_loss_1*a	    ஷ?    ஷ?      �?!    ஷ?)   ���?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   @�v?   @�v?      �?!   @�v?) )bڲ�<>2�T7��?�vV�R9?�������:              �?        ��/r�       ۞��	�L����AP*�
u
generator_loss_1*a	   @5E�?   @5E�?      �?!   @5E�?)����eh�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   `s�?   `s�?      �?!   `s�?) �Kg�*>2>h�'�?x?�x�?�������:              �?        �����       ۞��	�xo���Ax*�
u
generator_loss_1*a	    �f�?    �f�?      �?!    �f�?) H��*�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   �P6 ?   �P6 ?      �?!   �P6 ?) ���Ym>2>�?�s��>�FF�G ?�������:              �?        d��       S�	������A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �� �|{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) �Q�7�F>2��ڋ?�.�?�������:              �?        
���       S�	��{���A�*�
u
generator_loss_1*a	   @ж?   @ж?      �?!   @ж?)�8A�PC�?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �y^?   �y^?      �?!   �y^?)@����5>2��d�r?�5�i}1?�������:              �?        )}��       S�	����A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) Ȯ���?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   �:�?   �:�?      �?!   �:�?) �RD�D>2��ڋ?�.�?�������:              �?        =���       S�	�����A�*�
u
generator_loss_1*a	   �N/�?   �N/�?      �?!   �N/�?) �X�F̀?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	    A
?    A
?      �?!    A
?) �nQ�5%>2����?f�ʜ�7
?�������:              �?        ���       S�	+���A�*�
u
generator_loss_1*a	    <T�?    <T�?      �?!    <T�?) �p�7�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �Ó,>2��[�?1��a˲?�������:              �?        �A�       S�	3I����A�*�
u
generator_loss_1*a	   ��:�?   ��:�?      �?!   ��:�?)��\��܀?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) _~��F>2>�?�s��>�FF�G ?�������:              �?        �����       S�	)Vd���A�*�
u
generator_loss_1*a	    �@�?    �@�?      �?!    �@�?) ��>a�?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �e��>   �e��>      �?!   �e��>) ����>2��Zr[v�>O�ʗ��>�������:              �?        g����       S�	?����A�*�
u
generator_loss_1*a	    q�?    q�?      �?!    q�?) �=���?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	    
G?    
G?      �?!    
G?) 4�RA�N>2ji6�9�?�S�F !?�������:              �?        ��YR�       S�	����A�*�
u
generator_loss_1*a	   `���?   `���?      �?!   `���?)@��+h}?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   `�Q?   `�Q?      �?!   `�Q?) eU��)>2f�ʜ�7
?>h�'�?�������:              �?        �����       S�	&�Y���A�*�
u
generator_loss_1*a	   `&8�?   `&8�?      �?!   `&8�?) ~�ـ?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   @�F?   @�F?      �?!   @�F?) `���>2a�$��{E?
����G?�������:              �?        ���       S�	������A�*�
u
generator_loss_1*a	   ` �?   ` �?      �?!   ` �?) F��[�?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	    �w?    �w?      �?!    �w?) ���5!>26�]��?����?�������:              �?        ���{�       S�	������A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �
3|y�?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   @�?   @�?      �?!   @�?)��&�d!>26�]��?����?�������:              �?        rM0�       S�	��]���A�*�
u
generator_loss_1*a	   `6ظ?   `6ظ?      �?!   `6ظ?) elnJ�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   `��3?   `��3?      �?!   `��3?)@^��bx>2��82?�u�w74?�������:              �?        ��I�       S�	�Y���A�*�
u
generator_loss_1*a	   `���?   `���?      �?!   `���?)@��Dx�?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   @y?   @y?      �?!   @y?)����UI>2�.�?ji6�9�?�������:              �?        4M��       S�	������A�*�
u
generator_loss_1*a	   ��ϸ?   ��ϸ?      �?!   ��ϸ?) ��<�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) ���1>2x?�x�?��d�r?�������:              �?        +)�c�       S�	ވ���A�*�
u
generator_loss_1*a	   ��e�?   ��e�?      �?!   ��e�?) ��yZ?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	    ��K?    ��K?      �?!    ��K?)  ��wS�>2�qU���I?IcD���L?�������:              �?        `t~q�       S�	w�K���A�*�
u
generator_loss_1*a	   �O��?   �O��?      �?!   �O��?)���=�!�?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �f�!?   �f�!?      �?!   �f�!?) ��#JS>2�S�F !?�[^:��"?�������:              �?        [T���       S�	�e��A�*�
u
generator_loss_1*a	   �ɸ?   �ɸ?      �?!   �ɸ?) ��/�2�?28/�C�ַ?%g�cE9�?�������:              �?        
w
discriminator_loss*a	    37?    37?      �?!    37?)@\��7>2�5�i}1?�T7��?�������:              �?        ��N��       �{�	@���A*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@o���?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �q?   �q?      �?!   �q?)@��|C5>2��d�r?�5�i}1?�������:              �?        �&��       ۞��	m���A(*�
u
generator_loss_1*a	   �;��?   �;��?      �?!   �;��?) ��?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	    b�?    b�?      �?!    b�?)@���.�:>2�5�i}1?�T7��?�������:              �?        ��Li�       ۞��	�a��AP*�
u
generator_loss_1*a	   �EJ�?   �EJ�?      �?!   �EJ�?)@(��?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   @�K??   @�K??      �?!   @�K??)�����>2d�\D�X=?���#@?�������:              �?        ]�o�       ۞��	��1��Ax*�
u
generator_loss_1*a	   �%�?   �%�?      �?!   �%�?)@��s�{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    rp&?    rp&?      �?!    rp&?)@�Px_>2U�4@@�$?+A�F�&?�������:              �?        �S#�       S�	�K
��A�*�
u
generator_loss_1*a	    �̶?    �̶?      �?!    �̶?) �.�>�?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   `=?   `=?      �?!   `=?)@n��4>2��d�r?�5�i}1?�������:              �?        �%��       S�	<����A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@�ހ�"x?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��F?   ��F?      �?!   ��F?) ��|��*>2>h�'�?x?�x�?�������:              �?        v���       S�	� ���A�*�
u
generator_loss_1*a	   �8�?   �8�?      �?!   �8�?)@₝�~?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	    ?    ?      �?!    ?) �XMq�(>2f�ʜ�7
?>h�'�?�������:              �?        ����       S�	r���A�*�
u
generator_loss_1*a	   @L=�?   @L=�?      �?!   @L=�?) aӆ|�~?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) 1ƫ�>21��a˲?6�]��?�������:              �?        />p�       S�	%����A�*�
u
generator_loss_1*a	   @�E�?   @�E�?      �?!   @�E�?) auv�G|?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) Έ7i�)>2f�ʜ�7
?>h�'�?�������:              �?        �^��       S�	 ����A�*�
u
generator_loss_1*a	   �S{�?   �S{�?      �?!   �S{�?)�0��#;�?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �e�>   �e�>      �?!   �e�>) ��	�>2>�?�s��>�FF�G ?�������:              �?        VGfj�       S�	�m��A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) ��Ƽ<~?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �4?   �4?      �?!   �4?) r�]��C>2�vV�R9?��ڋ?�������:              �?        �*[�       S�	ɼ\��A�*�
u
generator_loss_1*a	   �a��?   �a��?      �?!   �a��?)@����z?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   `	(?   `	(?      �?!   `	(?) �>?#'b>2+A�F�&?I�I�)�(?�������:              �?        ��`#�       S�	+*T��A�*�
u
generator_loss_1*a	   `W]�?   `W]�?      �?!   `W]�?)@&���B?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �D�?   �D�?      �?!   �D�?) i��7>2�5�i}1?�T7��?�������:              �?        ԧ�i�       S�	i�I��A�*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?)@R?��v~?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   @?�%?   @?�%?      �?!   @?�%?) 	"T��\>2U�4@@�$?+A�F�&?�������:              �?        ���|�       S�	�H��A�*�
u
generator_loss_1*a	   ��X�?   ��X�?      �?!   ��X�?) !�y��y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    e$?    e$?      �?!    e$?) ��
�G>2��ڋ?�.�?�������:              �?        �p��       S�	AH��A�*�
u
generator_loss_1*a	   `� �?   `� �?      �?!   `� �?)@�(�}�{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �\�?   �\�?      �?!   �\�?) ��׵!>26�]��?����?�������:              �?        |���       S�	�L!��A�*�
u
generator_loss_1*a	    �O�?    �O�?      �?!    �O�?) ���c|?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    G�?    G�?      �?!    G�?) �����G>2��ڋ?�.�?�������:              �?        q�$8�       S�	zNe#��A�*�
u
generator_loss_1*a	   @�?   @�?      �?!   @�?)��=X��?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)��1�Qh#>2����?f�ʜ�7
?�������:              �?        A&f��       S�	?5l%��A�*�
u
generator_loss_1*a	   `�:�?   `�:�?      �?!   `�:�?)@��h�~?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) �$%s�>2>�?�s��>�FF�G ?�������:              �?        �b��       S�	2�~'��A�*�
u
generator_loss_1*a	    �^�?    �^�?      �?!    �^�?)@`ߎG?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   `å�>   `å�>      �?!   `å�>)@��2�I�=2pz�w�7�>I��P=�>�������:              �?        ը!l�       S�	ZI�)��A�*�
u
generator_loss_1*a	   �5�?   �5�?      �?!   �5�?) �"�Az?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    �`�>    �`�>      �?!    �`�>) ��*	>2O�ʗ��>>�?�s��>�������:              �?        ^'�#�       S�	?*�+��A�*�
u
generator_loss_1*a	   �B��?   �B��?      �?!   �B��?) d�T�`z?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   `@�?   `@�?      �?!   `@�?)@)�_1>2x?�x�?��d�r?�������:              �?        [5�       �{�	Es�-��A*�
u
generator_loss_1*a	   �g	�?   �g	�?      �?!   �g	�?) �Xߵ�{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?)  У=�">2����?f�ʜ�7
?�������:              �?        '*��       ۞��	���/��A(*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?)@���\~?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �� �>   �� �>      �?!   �� �>) 2C�u1>2I��P=�>��Zr[v�>�������:              �?        ӿ�@�       ۞��	ێ�1��AP*�
u
generator_loss_1*a	    �?    �?      �?!    �?)@,#��{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �sn���=2pz�w�7�>I��P=�>�������:              �?        �i!��       ۞��	E(4��Ax*�
u
generator_loss_1*a	   @F��?   @F��?      �?!   @F��?) q�yTz?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) ��6�">2����?f�ʜ�7
?�������:              �?        �,���       S�	PW86��A�*�
u
generator_loss_1*a	    g�?    g�?      �?!    g�?) ���:y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ��X�>   ��X�>      �?!   ��X�>)@�����=2pz�w�7�>I��P=�>�������:              �?        �{�       S�	��e8��A�*�
u
generator_loss_1*a	   �l�?   �l�?      �?!   �l�?) )-!�~?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   ��=?   ��=?      �?!   ��=?) �0�2>21��a˲?6�]��?�������:              �?        W�       S�	��:��A�*�
u
generator_loss_1*a	   �U��?   �U��?      �?!   �U��?)@:�P�z?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �$	?   �$	?      �?!   �$	?) Bxm�#>2����?f�ʜ�7
?�������:              �?        ��Q�       S�	��<��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) �\�5�w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `��?   `��?      �?!   `��?)@��;��>21��a˲?6�]��?�������:              �?        ]+}L�       S�	7��>��A�*�
u
generator_loss_1*a	   ��2�?   ��2�?      �?!   ��2�?) abN|?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) �Oh�p>2��[�?1��a˲?�������:              �?        �X�S�       S�	=�:A��A�*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?)@���H{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �?   �?      �?!   �?)@r�$V>26�]��?����?�������:              �?        �BA�       S�	�zC��A�*�
u
generator_loss_1*a	   @;b�?   @;b�?      �?!   @;b�?) i�6��y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �$��>   �$��>      �?!   �$��>) ~�e��=2�h���`�>�ߊ4F��>�������:              �?        b"��       S�	d��E��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) d}�/H{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	     ?     ?      �?!     ?)@`|�N,P>2ji6�9�?�S�F !?�������:              �?        �˯�       S�	S�H��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@<ۧEmz?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    "�?    "�?      �?!    "�?) @H=�p=>2�T7��?�vV�R9?�������:              �?        ����       S�	R�MJ��A�*�
u
generator_loss_1*a	    Z�?    Z�?      �?!    Z�?) @:k��{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) d��G��=2�ߊ4F��>})�l a�>�������:              �?        �E�K�       S�	ɔ�L��A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) 髳~~?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)   I>�=2})�l a�>pz�w�7�>�������:              �?        O��G�       S�	P��N��A�*�
u
generator_loss_1*a	    �Ҵ?    �Ҵ?      �?!    �Ҵ?) @��{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    �>    �>      �?!    �>) �:�1��=2��(���>a�Ϭ(�>�������:              �?        O�7��       S�	�@Q��A�*�
u
generator_loss_1*a	    �I�?    �I�?      �?!    �I�?)@�7�8@w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �6�>   �6�>      �?!   �6�>)@~��o�=2�h���`�>�ߊ4F��>�������:              �?        ��       S�	�șS��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) ��r�x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �TS?   �TS?      �?!   �TS?) ��0�>2��[�?1��a˲?�������:              �?        ����       S�	��U��A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) I���{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  F=�# >2I��P=�>��Zr[v�>�������:              �?        5��       S�	��UX��A�*�
u
generator_loss_1*a	   �8�?   �8�?      �?!   �8�?) IA��}?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	    `��>    `��>      �?!    `��>) �h�P�=2�f����>��(���>�������:              �?        ;�)��       S�		��Z��A�*�
u
generator_loss_1*a	    A�?    A�?      �?!    A�?) hI��{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �b��>   �b��>      �?!   �b��>) d>���=2�ߊ4F��>})�l a�>�������:              �?        �;�T�       S�	|4 ]��A�*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?)@�[�3;y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �l��>   �l��>      �?!   �l��>)@��Ƚ^�=2pz�w�7�>I��P=�>�������:              �?        vj>�       �{�	ɕq_��A*�
u
generator_loss_1*a	    [�?    [�?      �?!    [�?)  F�|?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �e2S�>2O�ʗ��>>�?�s��>�������:              �?        %�>��       ۞��	QF�a��A(*�
u
generator_loss_1*a	   �ݽ�?   �ݽ�?      �?!   �ݽ�?) d��v�z?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    U�)?    U�)?      �?!    U�)?) r��}d>2I�I�)�(?�7Kaa+?�������:              �?        ����       ۞��	 �Rd��AP*�
u
generator_loss_1*a	   ��̴?   ��̴?      �?!   ��̴?)@�=�	{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ��??   ��??      �?!   ��??)@�3�8>21��a˲?6�]��?�������:              �?        DK�E�       ۞��	�Q�f��Ax*�
u
generator_loss_1*a	   `kS�?   `kS�?      �?!   `kS�?)@����Ww?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@D6�m�3>2��d�r?�5�i}1?�������:              �?        7,�K�       S�	�MAi��A�*�
u
generator_loss_1*a	   ��9�?   ��9�?      �?!   ��9�?) $g�\�y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �� ?   �� ?      �?!   �� ?)@��^A>2��[�?1��a˲?�������:              �?        �(p=�       S�	w�k��A�*�
u
generator_loss_1*a	    qK�?    qK�?      �?!    qK�?)@ԫ�lW|?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ��� ?   ��� ?      �?!   ��� ?)@n#��>2�FF�G ?��[�?�������:              �?        p�F�       S�	�1?n��A�*�
u
generator_loss_1*a	   @I��?   @I��?      �?!   @I��?) Yo�/q}?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �b<�>   �b<�>      �?!   �b<�>)���Gq}>2>�?�s��>�FF�G ?�������:              �?        #^���       S�	�h�p��A�*�
u
generator_loss_1*a	   �	H�?   �	H�?      �?!   �	H�?) ��'_?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   ��0�>   ��0�>      �?!   ��0�>)������>2O�ʗ��>>�?�s��>�������:              �?        P����       S�	Z[Qs��A�*�
u
generator_loss_1*a	   `!0�?   `!0�?      �?!   `!0�?)@��d�|?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) N�_s�=2��(���>a�Ϭ(�>�������:              �?        o��       S�	7��u��A�*�
u
generator_loss_1*a	   �y�?   �y�?      �?!   �y�?) �ړG�x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �E�>   �E�>      �?!   �E�>) ࣶ4�=2�h���`�>�ߊ4F��>�������:              �?        ����       S�	m�tx��A�*�
u
generator_loss_1*a	   �^��?   �^��?      �?!   �^��?) ��?Ix?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) B4�>2��Zr[v�>O�ʗ��>�������:              �?        8�Jb�       S�	{��A�*�
u
generator_loss_1*a	    �R�?    �R�?      �?!    �R�?) ��k|?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   `�kY?   `�kY?      �?!   `�kY?) =Qc1�>2��bB�SY?�m9�H�[?�������:              �?        ���       S�	e��}��A�*�
u
generator_loss_1*a	   �t��?   �t��?      �?!   �t��?)@�t�Tz?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   @?�?   @?�?      �?!   @?�?) 	B�4Z8>2�5�i}1?�T7��?�������:              �?        ���n�       S�	�F8���A�*�
u
generator_loss_1*a	   �8�?   �8�?      �?!   �8�?)@|!���~?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   �r��>   �r��>      �?!   �r��>) ��E��=2�h���`�>�ߊ4F��>�������:              �?        �\�\�       S�	�ڂ��A�*�
u
generator_loss_1*a	   �%}�?   �%}�?      �?!   �%}�?) �G��w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��
?   ��
?      �?!   ��
?) h�ˮ&>2f�ʜ�7
?>h�'�?�������:              �?        A����       S�	o�|���A�*�
u
generator_loss_1*a	   @�Ѳ?   @�Ѳ?      �?!   @�Ѳ?) q��%"v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ĳ�>    ĳ�>      �?!    ĳ�>) ��>2��Zr[v�>O�ʗ��>�������:              �?        �}"V�       S�	wp"���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@��I��{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ����y>2>�?�s��>�FF�G ?�������:              �?        ���       S�	{q͊��A�*�
u
generator_loss_1*a	    ;��?    ;��?      �?!    ;��?)@|�|�{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �h��>   �h��>      �?!   �h��>) ɕ�d��=2pz�w�7�>I��P=�>�������:              �?        �b	�       S�	�[z���A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@~�ϊ�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��(?    ��(?      �?!    ��(?) �ӂgsc>2I�I�)�(?�7Kaa+?�������:              �?        7���       S�	]�(���A�*�
u
generator_loss_1*a	   ��v�?   ��v�?      �?!   ��v�?) �R?�w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    a%?    a%?      �?!    a%?) L�#�[>2U�4@@�$?+A�F�&?�������:              �?        U���       S�	�ْ��A�*�
u
generator_loss_1*a	   @Q�?   @Q�?      �?!   @Q�?) y] �f|?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   @$�?   @$�?      �?!   @$�?) !j���>21��a˲?6�]��?�������:              �?        ��7�       S�	�#����A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@�W-��}?2� l(��?8/�C�ַ?�������:              �?        
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@���q�>2�FF�G ?��[�?�������:              �?        ��u��       �{�	*C6���A*�
u
generator_loss_1*a	    w��?    w��?      �?!    w��?) uXւz?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   @��+?   @��+?      �?!   @��+?)�P�"h>2�7Kaa+?��VlQ.?�������:              �?        �w���       ۞��	���A(*�
u
generator_loss_1*a	   ��q�?   ��q�?      �?!   ��q�?) �n��z?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    (x?    (x?      �?!    (x?)  2�|6!>26�]��?����?�������:              �?        i�� �       ۞��	�^Ɲ��AP*�
u
generator_loss_1*a	   ��T�?   ��T�?      �?!   ��T�?) D�X��y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    �?    �?      �?!    �?) ����>21��a˲?6�]��?�������:              �?        ����       ۞��	v����Ax*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@�D��u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �?   �?      �?!   �?) ���%B>2�vV�R9?��ڋ?�������:              �?        :o%�       S�	�X���A�*�
u
generator_loss_1*a	   �s��?   �s��?      �?!   �s��?)@0�:�z?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) Hn6 >26�]��?����?�������:              �?        5r"�       S�	N*$���A�*�
u
generator_loss_1*a	   �lͳ?   �lͳ?      �?!   �lͳ?) )��/�x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �3R?   �3R?      �?!   �3R?) ą'�>21��a˲?6�]��?�������:              �?        �S���       S�	�����A�*�
u
generator_loss_1*a	    *��?    *��?      �?!    *��?) @n�i�x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �?   �?      �?!   �?)@�C>�>2�FF�G ?��[�?�������:              �?        �-#X�       S�	�ϫ��A�*�
u
generator_loss_1*a	   @ Ͳ?   @ Ͳ?      �?!   @ Ͳ?) h&�v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) y���9>2�5�i}1?�T7��?�������:              �?        F1���       S�	.դ���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ��@OWx?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    �c?    �c?      �?!    �c?)   �{�,>2>h�'�?x?�x�?�������:              �?        �����       S�	I�����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@؍�6"}?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    ,"�>    ,"�>      �?!    ,"�>) � :7`�=28K�ߝ�>�h���`�>�������:              �?        V�$�       S�	��l���A�*�
u
generator_loss_1*a	   ��V�?   ��V�?      �?!   ��V�?) ��}�_w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �8�?   �8�?      �?!   �8�?)��0���I>2�.�?ji6�9�?�������:              �?        ʽ�Y�       S�	5�S���A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �	قdv?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `� ?   `� ?      �?!   `� ?)@fQ&P>2ji6�9�?�S�F !?�������:              �?        w۔�       S�	�<���A�*�
u
generator_loss_1*a	   ੱ�?   ੱ�?      �?!   ੱ�?)@��9�=x?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `u�?   `u�?      �?!   `u�?)@�-9�1>2x?�x�?��d�r?�������:              �?        -4��       S�	n�&���A�*�
u
generator_loss_1*a	    �:�?    �:�?      �?!    �:�?) �;�w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @Vu?   @Vu?      �?!   @Vu?)�x����">2����?f�ʜ�7
?�������:              �?        �Vձ�       S�	�����A�*�
u
generator_loss_1*a	   �5��?   �5��?      �?!   �5��?)@hE�x?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) �z��>26�]��?����?�������:              �?        U�g�       S�	WN���A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) �?��y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    �?    �?      �?!    �?) 9c�VD>2�vV�R9?��ڋ?�������:              �?        %;���       S�	p}����A�*�
u
generator_loss_1*a	   `WZ�?   `WZ�?      �?!   `WZ�?)@&Ո��y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ��o?   ��o?      �?!   ��o?)@�z�>>2��[�?1��a˲?�������:              �?        Gv7��       S�	������A�*�
u
generator_loss_1*a	    `ɳ?    `ɳ?      �?!    `ɳ?)  @~*xx?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �W��>   �W��>      �?!   �W��>) B+Y=>2��Zr[v�>O�ʗ��>�������:              �?        �Rë�       S�	R����A�*�
u
generator_loss_1*a	   @aײ?   @aײ?      �?!   @aײ?) ��/v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �q�?   �q�?      �?!   �q�?)�Xd��F>2��ڋ?�.�?�������:              �?        I8��       S�	������A�*�
u
generator_loss_1*a	   @>9�?   @>9�?      �?!   @>9�?) 1z4'|?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    �~�>    �~�>      �?!    �~�>) HD4ߟ>2O�ʗ��>>�?�s��>�������:              �?        -`Y�       S�	� ���A�*�
u
generator_loss_1*a	   �?   �?      �?!   �?) $3��zu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    �n�>    �n�>      �?!    �n�>)  @T��=2�f����>��(���>�������:              �?        ���*�       S�	�����A�*�
u
generator_loss_1*a	    0q�?    0q�?      �?!    0q�?)  ��z?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �^�>   �^�>      �?!   �^�>) ������=28K�ߝ�>�h���`�>�������:              �?        �V���       �{�	� ���A*�
u
generator_loss_1*a	   ��ش?   ��ش?      �?!   ��ش?) ��Y�){?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �I~�>   �I~�>      �?!   �I~�>) ��Jx �=2�h���`�>�ߊ4F��>�������:              �?        2Wo��       ۞��	�*���A(*�
u
generator_loss_1*a	   `�U�?   `�U�?      �?!   `�U�?)@�����y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   @.|�>   @.|�>      �?!   @.|�>)�زbd<�=2��(���>a�Ϭ(�>�������:              �?        _�G�       ۞��	�k.���AP*�
u
generator_loss_1*a	   �2��?   �2��?      �?!   �2��?)@.0�T�w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �xn�>   �xn�>      �?!   �xn�>) ��Z��=2a�Ϭ(�>8K�ߝ�>�������:              �?         ��{�       ۞��	�QB���Ax*�
u
generator_loss_1*a	   @A��?   @A��?      �?!   @A��?) �z�us?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �^?    �^?      �?!    �^?) @T��>2��[�?1��a˲?�������:              �?        {����       S�	��d���A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) ����bx?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   @�S?   @�S?      �?!   @�S?) ��ps��>2�lDZrS?<DKc��T?�������:              �?        [�ч�       S�	������A�*�
u
generator_loss_1*a	   �'�?   �'�?      �?!   �'�?)@"����v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�Ȏ ��>2O�ʗ��>>�?�s��>�������:              �?         �L��       S�	�v����A�*�
u
generator_loss_1*a	   �A�?   �A�?      �?!   �A�?) 1�kbUv?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@r�>I�=2pz�w�7�>I��P=�>�������:              �?        @�0B�       S�	�v����A�*�
u
generator_loss_1*a	   �G�?   �G�?      �?!   �G�?) q�:w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��	?   ��	?      �?!   ��	?) ����$>2����?f�ʜ�7
?�������:              �?        &����       S�	�����A�*�
u
generator_loss_1*a	   �.-�?   �.-�?      �?!   �.-�?)@��R��v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) @���;�=2�ߊ4F��>})�l a�>�������:              �?        ¯[��       S�	mE���A�*�
u
generator_loss_1*a	   `�}�?   `�}�?      �?!   `�}�?)@N��=z?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �z�>   �z�>      �?!   �z�>)@���eG�=2�uE����>�f����>�������:              �?          �       S�	Ӗq���A�*�
u
generator_loss_1*a	    �б?    �б?      �?!    �б?) ��|�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �wf�>   �wf�>      �?!   �wf�>)� (A��=2��(���>a�Ϭ(�>�������:              �?        i�M��       S�	�]����A�*�
u
generator_loss_1*a	    bM�?    bM�?      �?!    bM�?) @�?Iw?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ">?    ">?      �?!    ">?) d`���@>2�vV�R9?��ڋ?�������:              �?        Q���       S�	5w����A�*�
u
generator_loss_1*a	   `R��?   `R��?      �?!   `R��?)@D:�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �w[?   �w[?      �?!   �w[?)@ �Gk>2��[�?1��a˲?�������:              �?        ����       S�	ɿ(��A�*�
u
generator_loss_1*a	   ��V�?   ��V�?      �?!   ��V�?) ��Lu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �����=2a�Ϭ(�>8K�ߝ�>�������:              �?        ���"�       S�	�Pe��A�*�
u
generator_loss_1*a	   `���?   `���?      �?!   `���?)@����x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �n�>   �n�>      �?!   �n�>)@ F����=2�ߊ4F��>})�l a�>�������:              �?        �����       S�	[$���A�*�
u
generator_loss_1*a	   �e�?   �e�?      �?!   �e�?)@�IT5�y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   `�g�>   `�g�>      �?!   `�g�>)@&G82`�=2�f����>��(���>�������:              �?        רL��       S�	5b�
��A�*�
u
generator_loss_1*a	   @cu�?   @cu�?      �?!   @cu�?) ���(�w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    �D%?    �D%?      �?!    �D%?) ��=�E\>2U�4@@�$?+A�F�&?�������:              �?        9����       S�	K4;��A�*�
u
generator_loss_1*a	    6ò?    6ò?      �?!    6ò?) @�4� v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `GU�>   `GU�>      �?!   `GU�>)@f�P\�=2})�l a�>pz�w�7�>�������:              �?        ����       S�	�A���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ��\�v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `.��>   `.��>      �?!   `.��>)@ja��=2�f����>��(���>�������:              �?        _yR!�       S�	?����A�*�
u
generator_loss_1*a	   @�$�?   @�$�?      �?!   @�$�?) �J�[y?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   @6��>   @6��>      �?!   @6��>)���#���=2E��a�W�>�ѩ�-�>�������:              �?         �U��       S�	g$/��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) $�)��w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �!sԂ�=2�f����>��(���>�������:              �?        �y�W�       S�	J@���A�*�
u
generator_loss_1*a	   `2-�?   `2-�?      �?!   `2-�?)@��O��v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) @Aڛ��=2a�Ϭ(�>8K�ߝ�>�������:              �?        ׸V�       �{�	����A*�
u
generator_loss_1*a	   @8|�?   @8|�?      �?!   @8|�?) ���s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    /�>    /�>      �?!    /�>)@����=2pz�w�7�>I��P=�>�������:              �?        ���9�       ۞��	�nV"��A(*�
u
generator_loss_1*a	    l��?    l��?      �?!    l��?)@���x?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �\�?   �\�?      �?!   �\�?)@ �>21��a˲?6�]��?�������:              �?        �\mC�       ۞��	<!�%��AP*�
u
generator_loss_1*a	   �9��?   �9��?      �?!   �9��?)@X�c��w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `#2�>   `#2�>      �?!   `#2�>)@6f�l��=2�ߊ4F��>})�l a�>�������:              �?        ���       ۞��	��))��Ax*�
u
generator_loss_1*a	   ��6�?   ��6�?      �?!   ��6�?) YG`w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    �͑?    �͑?      �?!    �͑?) ����3?2���&�?�Rc�ݒ?�������:              �?        Oe�C�       S�	��,��A�*�
u
generator_loss_1*a	   @�f�?   @�f�?      �?!   @�f�?) ���ņw?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��??   ��??      �?!   ��??) �!&Š>21��a˲?6�]��?�������:              �?        �,�:�       S�	��0��A�*�
u
generator_loss_1*a	   `F �?   `F �?      �?!   `F �?)@�5%��v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    /Y�>    /Y�>      �?!    /Y�>) fg"�>2��Zr[v�>O�ʗ��>�������:              �?        E|E��       S�	*P�3��A�*�
u
generator_loss_1*a	    c��?    c��?      �?!    c��?) �D�vpu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `,3�>   `,3�>      �?!   `,3�>)@_-
�=2�uE����>�f����>�������:              �?        ����       S�	D��6��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@ri��z?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    �9�>    �9�>      �?!    �9�>)@D>���=2���%�>�uE����>�������:              �?        0i1�       S�	��~:��A�*�
u
generator_loss_1*a	   @�]�?   @�]�?      �?!   @�]�?) �}0�pw?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �*��>   �*��>      �?!   �*��>)@��[�t�=2���%�>�uE����>�������:              �?        ��Ý�       S�	5q >��A�*�
u
generator_loss_1*a	   ��p�?   ��p�?      �?!   ��p�?) D��	Au?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @P�>   @P�>      �?!   @P�>)�::�P�=2�iD*L��>E��a�W�>�������:              �?        8�|��       S�	u-�A��A�*�
u
generator_loss_1*a	    Y�?    Y�?      �?!    Y�?) /þ�v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��NN?   ��NN?      �?!   ��NN?) ������>2IcD���L?k�1^�sO?�������:              �?        :\(�       S�	��E��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@�}\�v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) a�F^��=2pz�w�7�>I��P=�>�������:              �?        0�|d�       S�	R��H��A�*�
u
generator_loss_1*a	   @�ǲ?   @�ǲ?      �?!   @�ǲ?) !N��
v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    EI�>    EI�>      �?!    EI�>) �ds���=2�h���`�>�ߊ4F��>�������:              �?        �;x9�       S�	ܷ'L��A�*�
u
generator_loss_1*a	   �2'�?   �2'�?      �?!   �2'�?) d��w�v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��F�>   ��F�>      �?!   ��F�>) �~}B��=2a�Ϭ(�>8K�ߝ�>�������:              �?        V��3�       S�	85�O��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@�ԕu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��-�>   ��-�>      �?!   ��-�>)�lη9��=28K�ߝ�>�h���`�>�������:              �?        ݴ��       S�	��ZS��A�*�
u
generator_loss_1*a	    G�?    G�?      �?!    G�?) �!^�x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �B�>   �B�>      �?!   �B�>) y�g���=2�f����>��(���>�������:              �?        ��6�       S�	�W��A�*�
u
generator_loss_1*a	    IӲ?    IӲ?      �?!    IӲ?)@4��J&v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �L��>   �L��>      �?!   �L��>)@\Q�?��=2})�l a�>pz�w�7�>�������:              �?        E��t�       S�	;��Z��A�*�
u
generator_loss_1*a	    n��?    n��?      �?!    n��?) @��~v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �M�>   �M�>      �?!   �M�>)@�n��=2�h���`�>�ߊ4F��>�������:              �?        `���       S�	�P^��A�*�
u
generator_loss_1*a	   ��г?   ��г?      �?!   ��г?)@�4q��x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �Ֆ��=2�_�T�l�>�iD*L��>�������:              �?        ��{�       S�	Y�a��A�*�
u
generator_loss_1*a	   ��K�?   ��K�?      �?!   ��K�?) �rA�Ew?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    %�>    %�>      �?!    %�>) b_��7�=2�_�T�l�>�iD*L��>�������:              �?        3���       S�	�G�e��A�*�
u
generator_loss_1*a	   @ϲ?   @ϲ?      �?!   @ϲ?) )/Rcv?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �<��>   �<��>      �?!   �<��>) ���&��=2E��a�W�>�ѩ�-�>�������:              �?        �p���       S�	��^i��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) dF��x?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @Δ ?   @Δ ?      �?!   @Δ ?) �r/Q>2ji6�9�?�S�F !?�������:              �?        R��       �{�	m��A*�
u
generator_loss_1*a	    �t�?    �t�?      �?!    �t�?)@0�e��w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    p�>    p�>      �?!    p�>) �����=2�ѩ�-�>���%�>�������:              �?        �Ҽ>�       ۞��	[�p��A(*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) ���Dx?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) 㷽�=2['�?��>K+�E���>�������:              �?        ]_�       ۞��	`�ut��AP*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@����x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    �j�>    �j�>      �?!    �j�>)@���g�=2��>M|K�>�_�T�l�>�������:              �?        !���       ۞��	Qi5x��Ax*�
u
generator_loss_1*a	   �w	�?   �w	�?      �?!   �w	�?)@b�ߨ{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) D��Ҙ=2�XQ��>�����>�������:              �?        �Q��       S�	F��{��A�*�
u
generator_loss_1*a	   @ȑ�?   @ȑ�?      �?!   @ȑ�?) AR�k�w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �O��>   �O��>      �?!   �O��>) �-�[�=2�XQ��>�����>�������:              �?        q?���       S�	jm���A�*�
u
generator_loss_1*a	   �V\�?   �V\�?      �?!   �V\�?) ��Tbmw?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �ea�>   �ea�>      �?!   �ea�>)������=2�ѩ�-�>���%�>�������:              �?        �j�U�       S�	��x���A�*�
u
generator_loss_1*a	    i��?    i��?      �?!    i��?) ���u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@�B A�=2�ѩ�-�>���%�>�������:              �?        ��"p�       S�	�L���A�*�
u
generator_loss_1*a	   @(I�?   @(I�?      �?!   @(I�?) A�?w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    4��>    4��>      �?!    4��>) �6�%?�=2��(���>a�Ϭ(�>�������:              �?        $�L`�       S�	d���A�*�
u
generator_loss_1*a	   �	��?   �	��?      �?!   �	��?) �%��J}?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ���c>   ���c>      �?!   ���c>) �O��]�<2�����0c>cR�k�e>�������:              �?        [���       S�	�����A�*�
u
generator_loss_1*a	    S��?    S��?      �?!    S��?)@�W��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) �y[�>2��[�?1��a˲?�������:              �?        @^���       S�	|޽���A�*�
u
generator_loss_1*a	    9�?    9�?      �?!    9�?)@�w�Wkt?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)��x5f>2��Zr[v�>O�ʗ��>�������:              �?        |(��       S�	1����A�*�
u
generator_loss_1*a	    �ձ?    �ձ?      �?!    �ձ?)@�����s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �~?   �~?      �?!   �~?)@*�M)>21��a˲?6�]��?�������:              �?        4Qr�       S�	a%w���A�*�
u
generator_loss_1*a	   @䒳?   @䒳?      �?!   @䒳?) !H�"�w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    7�?    7�?      �?!    7�?)@��a^�8>2�5�i}1?�T7��?�������:              �?        ���s�       S�	�~f���A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) FHSx?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) a�_��>21��a˲?6�]��?�������:              �?        5����       S�	��N���A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) �\�w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ʶ�>    ʶ�>      �?!    ʶ�>) �(�>2��Zr[v�>O�ʗ��>�������:              �?        ��8��       S�	Cs8���A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)  9tat?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)�|��H@>2�T7��?�vV�R9?�������:              �?        �P��       S�	8����A�*�
u
generator_loss_1*a	   `9ɲ?   `9ɲ?      �?!   `9ɲ?)@�Yʣv?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ���>2>�?�s��>�FF�G ?�������:              �?        `�
'�       S�	�����A�*�
u
generator_loss_1*a	   �ڋ�?   �ڋ�?      �?!   �ڋ�?) �v�{q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��l?   ��l?      �?!   ��l?)�55�%A>2�vV�R9?��ڋ?�������:              �?        ����       S�	�����A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@0w�Zt?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) 2n ->2>h�'�?x?�x�?�������:              �?        �o�@�       S�	�-���A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) ��'�x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ��$?   ��$?      �?!   ��$?)@��ň`Z>2�[^:��"?U�4@@�$?�������:              �?        ڕ�0�       S�	�m���A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@H&_F=x?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @eU ?   @eU ?      �?!   @eU ?) �HF��>2�FF�G ?��[�?�������:              �?        ��}�       S�	��
���A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �~�$x?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��� ?   ��� ?      �?!   ��� ?)@胕OC>2�FF�G ?��[�?�������:              �?        ����       �{�	�&����A*�
u
generator_loss_1*a	    �7�?    �7�?      �?!    �7�?)  !�1�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �?    �?      �?!    �?)@d���90>2x?�x�?��d�r?�������:              �?        ���       ۞��	Y_���A(*�
u
generator_loss_1*a	   �˲?   �˲?      �?!   �˲?) H�v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ���=2�f����>��(���>�������:              �?        áB��       ۞��	�����AP*�
u
generator_loss_1*a	    Q�?    Q�?      �?!    Q�?) @j��t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `�0�>   `�0�>      �?!   `�0�>)@��y��=2pz�w�7�>I��P=�>�������:              �?        �f�w�       ۞��	7.���Ax*�
u
generator_loss_1*a	   �M�?   �M�?      �?!   �M�?) dǭh�{?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  �����=2�ߊ4F��>})�l a�>�������:              �?        ��(��       S�	��@���A�*�
u
generator_loss_1*a	    �?�?    �?�?      �?!    �?�?) 4j(w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@|�w��=2�ߊ4F��>})�l a�>�������:              �?        �k<��       S�	�T���A�*�
u
generator_loss_1*a	   �<�?   �<�?      �?!   �<�?) ���t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    �?    �?      �?!    �?) ��:��A>2�vV�R9?��ڋ?�������:              �?        E���       S�	W�k���A�*�
u
generator_loss_1*a	   @�h�?   @�h�?      �?!   @�h�?) ѓ��w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @�]?   @�]?      �?!   @�]?) �Vgg�>21��a˲?6�]��?�������:              �?        9�|�       S�	������A�*�
u
generator_loss_1*a	   `�߱?   `�߱?      �?!   `�߱?)@����s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�"�>   @�"�>      �?!   @�"�>)�<���K>2>�?�s��>�FF�G ?�������:              �?        �tK|�       S�	�׺���A�*�
u
generator_loss_1*a	    �ܲ?    �ܲ?      �?!    �ܲ?)@D� =v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �¹�>   �¹�>      �?!   �¹�>) ��aK�	>2O�ʗ��>>�?�s��>�������:              �?        �W��       S�	ߡ����A�*�
u
generator_loss_1*a	   @&I�?   @&I�?      �?!   @&I�?) qc}��t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) r�F�=28K�ߝ�>�h���`�>�������:              �?        �<��       S�	t�	���A�*�
u
generator_loss_1*a	   �	ʱ?   �	ʱ?      �?!   �	ʱ?)@��K�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�x���0�=2�h���`�>�ߊ4F��>�������:              �?        A��d�       S�	u
8���A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) ͼ�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @I��>   @I��>      �?!   @I��>)���҆�>2>�?�s��>�FF�G ?�������:              �?        ��Ѿ�       S�	�j���A�*�
u
generator_loss_1*a	    J;�?    J;�?      �?!    J;�?)@h|B�t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �X��>   �X��>      �?!   �X��>)@�� ���=2pz�w�7�>I��P=�>�������:              �?        d��       S�	F ����A�*�
u
generator_loss_1*a	   �X��?   �X��?      �?!   �X��?)@�*���w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)���Z��=2a�Ϭ(�>8K�ߝ�>�������:              �?        ��       S�	������A�*�
u
generator_loss_1*a	   ��Ѳ?   ��Ѳ?      �?!   ��Ѳ?)@�2"v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @�?   @�?      �?!   @�?)�T�w�b/>2>h�'�?x?�x�?�������:              �?        !S�+�       S�	*( 	��A�*�
u
generator_loss_1*a	   @]�?   @]�?      �?!   @]�?) yW���x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �d��>   �d��>      �?!   �d��>)@�{C��=2pz�w�7�>I��P=�>�������:              �?        挻%�       S�	�I	��A�*�
u
generator_loss_1*a	   ��]�?   ��]�?      �?!   ��]�?)@.N��u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) )����=2�ߊ4F��>})�l a�>�������:              �?        �8q%�       S�	�ӈ	��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@�b�B�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `"t�>   `"t�>      �?!   `"t�>)@ڹ����=2�h���`�>�ߊ4F��>�������:              �?        �H	��       S�	��	��A�*�
u
generator_loss_1*a	   ँ�?   ँ�?      �?!   ँ�?)@�>՘'s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) }=�">2��Zr[v�>O�ʗ��>�������:              �?        ��F�       S�	0e	��A�*�
u
generator_loss_1*a	   @�A�?   @�A�?      �?!   @�A�?) y���,w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �]8�>   �]8�>      �?!   �]8�>) 2Q8�{�=2a�Ϭ(�>8K�ߝ�>�������:              �?        yJ��       S�	`	��A�*�
u
generator_loss_1*a	   `䘱?   `䘱?      �?!   `䘱?)@��K�Zs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �'��>   �'��>      �?!   �'��>) �ա��=2��(���>a�Ϭ(�>�������:              �?        C��g�       S�	(׼	��A�*�
u
generator_loss_1*a	   @P��?   @P��?      �?!   @P��?) ���$ms?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��p�>   ��p�>      �?!   ��p�>) r]��>2>�?�s��>�FF�G ?�������:              �?        4�N�       �{�	Rd	��A*�
u
generator_loss_1*a	    �?    �?      �?!    �?) �[�lr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �7p�>   �7p�>      �?!   �7p�>) ��W��=28K�ߝ�>�h���`�>�������:              �?        �t�i�       ۞��	=�{"	��A(*�
u
generator_loss_1*a	   �ɩ�?   �ɩ�?      �?!   �ɩ�?) ���u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    0ݗ>    0ݗ>      �?!    0ݗ>) `��A=2.��fc��>39W$:��>�������:              �?        ��R�       ۞��	`��&	��AP*�
u
generator_loss_1*a	   �~�?   �~�?      �?!   �~�?)@�+Sr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)��]�k=2���?�ګ>����>�������:              �?        $'���       ۞��	W�,+	��Ax*�
u
generator_loss_1*a	   @n��?   @n��?      �?!   @n��?) ��E�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) A���l�=2�uE����>�f����>�������:              �?        飶5�       S�	<ޏ/	��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@�˖Mr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `�h?   `�h?      �?!   `�h?) ��m' !>26�]��?����?�������:              �?        ��� �       S�	��3	��A�*�
u
generator_loss_1*a	   ��|�?   ��|�?      �?!   ��|�?) ��\u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    �3&?    �3&?      �?!    �3&?) @*�`�^>2U�4@@�$?+A�F�&?�������:              �?        5�J�       S�	.Yn8	��A�*�
u
generator_loss_1*a	   @�o�?   @�o�?      �?!   @�o�?) )�E��w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) �8�v�A>2�vV�R9?��ڋ?�������:              �?        �Y��       S�	�Z�<	��A�*�
u
generator_loss_1*a	   @9ΰ?   @9ΰ?      �?!   @9ΰ?) ���Ԧq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���!?   ���!?      �?!   ���!?) Ā8��S>2�S�F !?�[^:��"?�������:              �?        �����       S�	��ZA	��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@��ċ�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���(?   ���(?      �?!   ���(?)�\��BCc>2+A�F�&?I�I�)�(?�������:              �?        ����       S�	���E	��A�*�
u
generator_loss_1*a	   ��y�?   ��y�?      �?!   ��y�?) �I{�Uu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �Q;?   �Q;?      �?!   �Q;?) $��'�9>2�5�i}1?�T7��?�������:              �?        �4O��       S�	sxbJ	��A�*�
u
generator_loss_1*a	    �?    �?      �?!    �?) �
B�v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	     �?     �?      �?!     �?)  @�P�>21��a˲?6�]��?�������:              �?        ����       S�	�N	��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) ���ds?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    f�:?    f�:?      �?!    f�:?) �� t)�>2uܬ�@8?��%>��:?�������:              �?        �� �       S�	4InS	��A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) Y��x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) R)k�A>2�vV�R9?��ڋ?�������:              �?        %n�1�       S�	��W	��A�*�
u
generator_loss_1*a	   �"!�?   �"!�?      �?!   �"!�?) d�>Ҋt?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �C!?   �C!?      �?!   �C!?) ��&�� >26�]��?����?�������:              �?        ���       S�	�D�\	��A�*�
u
generator_loss_1*a	    P�?    P�?      �?!    P�?)  �+t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `p�>   `p�>      �?!   `p�>)@BM�DB�=2})�l a�>pz�w�7�>�������:              �?        6��%�       S�	9�a	��A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) �8���t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �i/?   �i/?      �?!   �i/?) ��>w9>2�5�i}1?�T7��?�������:              �?        �Y�:�       S�	k��e	��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �]���s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �P?    �P?      �?!    �P?)@L|�@/�>2k�1^�sO?nK���LQ?�������:              �?        9����       S�	r�Qj	��A�*�
u
generator_loss_1*a	   �S��?   �S��?      �?!   �S��?)@�C��u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @�$?   @�$?      �?!   @�$?) �����Z>2U�4@@�$?+A�F�&?�������:              �?        �����       S�	��n	��A�*�
u
generator_loss_1*a	   �Y��?   �Y��?      �?!   �Y��?) qg��w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    !5?    !5?      �?!    !5?) J� "G>2��ڋ?�.�?�������:              �?        �-b��       S�	��s	��A�*�
u
generator_loss_1*a	   �(V�?   �(V�?      �?!   �(V�?)@&]�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) ��C�>>2��[�?1��a˲?�������:              �?        ����       S�	�{1x	��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@HUQ+=t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    N�	?    N�	?      �?!    N�	?) �Z��$>2����?f�ʜ�7
?�������:              �?        E]�f�       S�	S��|	��A�*�
u
generator_loss_1*a	    �$�?    �$�?      �?!    �$�?)@0�l��v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    �d?    �d?      �?!    �d?)@�I��>2��[�?1��a˲?�������:              �?        �E0��       �{�	��v�	��A*�
u
generator_loss_1*a	   `H��?   `H��?      �?!   `H��?)@b��Wx?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ������=2})�l a�>pz�w�7�>�������:              �?        �y�       ۞��	N�;�	��A(*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@�����u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@��*��=2���%�>�uE����>�������:              �?        ��%��       ۞��	�l��	��AP*�
u
generator_loss_1*a	   ��]�?   ��]�?      �?!   ��]�?)@�a�!�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �#-?   �#-?      �?!   �#-?) 8��tL>2�.�?ji6�9�?�������:              �?        ��t�       ۞��	̇Ï	��Ax*�
u
generator_loss_1*a	   @�2�?   @�2�?      �?!   @�2�?) ��Ѳt?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)���n�:>2>�?�s��>�FF�G ?�������:              �?        h\y�       S�	��~�	��A�*�
u
generator_loss_1*a	    �n�?    �n�?      �?!    �n�?)@$I#�<u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �0B!?   �0B!?      �?!   �0B!?) �ฝR>2�S�F !?�[^:��"?�������:              �?        l~R�       S�	5�4�	��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) !�Y޲u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �J6�>   �J6�>      �?!   �J6�>) �`�x>2��Zr[v�>O�ʗ��>�������:              �?        k���       S�	O��	��A�*�
u
generator_loss_1*a	   �k�?   �k�?      �?!   �k�?)@�#&�Nv?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��J�>   ��J�>      �?!   ��J�>) i��S�=2�f����>��(���>�������:              �?        Mɽ��       S�	q簢	��A�*�
u
generator_loss_1*a	   ��y�?   ��y�?      �?!   ��y�?) �pWs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@��mJ,�=2pz�w�7�>I��P=�>�������:              �?        ���       S�	%ͅ�	��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) @���q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	     ��>     ��>      �?!     ��>)@���Rh�=2�ߊ4F��>})�l a�>�������:              �?        Sz�L�       S�	��P�	��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)  @.�!t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) Lr��~�=28K�ߝ�>�h���`�>�������:              �?        � ��       S�	yU#�	��A�*�
u
generator_loss_1*a	   �$̱?   �$̱?      �?!   �$̱?) it���s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) 3+p-��=2a�Ϭ(�>8K�ߝ�>�������:              �?        u[�       S�	�c��	��A�*�
u
generator_loss_1*a	   �=±?   �=±?      �?!   �=±?) Q^R��s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) n�����=2a�Ϭ(�>8K�ߝ�>�������:              �?        ؿ�4�       S�	\�ɺ	��A�*�
u
generator_loss_1*a	   �4��?   �4��?      �?!   �4��?)@�^Etv?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)���7B+�=2�iD*L��>E��a�W�>�������:              �?        i�gA�       S�	����	��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �qLFs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �����=2�h���`�>�ߊ4F��>�������:              �?        �q%�       S�	ܖq�	��A�*�
u
generator_loss_1*a	   �R��?   �R��?      �?!   �R��?) �c!Qpu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �_��>2��Zr[v�>O�ʗ��>�������:              �?        �M�j�       S�	�zW�	��A�*�
u
generator_loss_1*a	   �A:�?   �A:�?      �?!   �A:�?)@8�V��t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ��¥��=2�ߊ4F��>})�l a�>�������:              �?        ���       S�	E�2�	��A�*�
u
generator_loss_1*a	    J�?    J�?      �?!    J�?)@䃋�t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �O5�>   �O5�>      �?!   �O5�>) ;����=2���%�>�uE����>�������:              �?        @���       S�	�4#�	��A�*�
u
generator_loss_1*a	   @ɖ�?   @ɖ�?      �?!   @ɖ�?) YS�јu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `,j�>   `,j�>      �?!   `,j�>) �9֢��=2�_�T�l�>�iD*L��>�������:              �?        �'D��       S�	r�	��A�*�
u
generator_loss_1*a	   �@D�?   �@D�?      �?!   �@D�?) 	f״�t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��#�>   ��#�>      �?!   ��#�>) ¥�r5�=2
�/eq
�>;�"�q�>�������:              �?        �(|�       S�	�	��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@�[�s�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��e�>   ��e�>      �?!   ��e�>)�p����=2E��a�W�>�ѩ�-�>�������:              �?        S|�N�       S�	@���	��A�*�
u
generator_loss_1*a	   �6��?   �6��?      �?!   �6��?) Y+���w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) �4�_�=2��>M|K�>�_�T�l�>�������:              �?        :�x��       S�	%��	��A�*�
u
generator_loss_1*a	    o{�?    o{�?      �?!    o{�?) �s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �p.�-�=2�iD*L��>E��a�W�>�������:              �?        �C{\�       �{�	=���	��A*�
u
generator_loss_1*a	    ݫ�?    ݫ�?      �?!    ݫ�?) ��O��u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �&��>   �&��>      �?!   �&��>)@t�Ҟ'�=2�ߊ4F��>})�l a�>�������:              �?        {Ar,�       ۞��	����	��A(*�
u
generator_loss_1*a	   ��~�?   ��~�?      �?!   ��~�?)@���}q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �t��>   �t��>      �?!   �t��>) "�K���=28K�ߝ�>�h���`�>�������:              �?        ��I��       ۞��	ʀ��	��AP*�
u
generator_loss_1*a	    kz�?    kz�?      �?!    kz�?)@<u�Wu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) `�i��=2��>M|K�>�_�T�l�>�������:              �?         ߄��       ۞��	m���	��Ax*�
u
generator_loss_1*a	   �|�?   �|�?      �?!   �|�?)@
Yys?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �9�>   �9�>      �?!   �9�>) ���u��=2pz�w�7�>I��P=�>�������:              �?        �d�N�       S�	���	��A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?) ��\�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?)  ![b>21��a˲?6�]��?�������:              �?        ��?�       S�	"4�
��A�*�
u
generator_loss_1*a	    ϱ?    ϱ?      �?!    ϱ?)@��]�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �Z�>   �Z�>      �?!   �Z�>) �7���=2��(���>a�Ϭ(�>�������:              �?        �s�Y�       S�	�~�	
��A�*�
u
generator_loss_1*a	    ᒱ?    ᒱ?      �?!    ᒱ?)@��bMs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �]��>   �]��>      �?!   �]��>)@�~4��=2�f����>��(���>�������:              �?        nOx�       S�	:�
��A�*�
u
generator_loss_1*a	   `�ٲ?   `�ٲ?      �?!   `�ٲ?)@F	�5v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �{��>   �{��>      �?!   �{��>)����L4�=2��>M|K�>�_�T�l�>�������:              �?        ��<9�       S�	}N*
��A�*�
u
generator_loss_1*a	   `�?   `�?      �?!   `�?)@&*��t?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) Re���!>2����?f�ʜ�7
?�������:              �?        C���       S�	}�C
��A�*�
u
generator_loss_1*a	   �� �?   �� �?      �?!   �� �?) �\�Ur?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �r�?   �r�?      �?!   �r�?) d��&�>26�]��?����?�������:              �?        X9��       S�	M�b
��A�*�
u
generator_loss_1*a	   ��<�?   ��<�?      �?!   ��<�?)@2�;��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) i���=2�uE����>�f����>�������:              �?        [t� �       S�	�0�#
��A�*�
u
generator_loss_1*a	   �"Ȱ?   �"Ȱ?      �?!   �"Ȱ?)@���q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `�?   `�?      �?!   `�?)@N4J��?>2�T7��?�vV�R9?�������:              �?        ����       S�	��(
��A�*�
u
generator_loss_1*a	   �\�?   �\�?      �?!   �\�?)@�'TJr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �uW�>   �uW�>      �?!   �uW�>) r�H��=2��(���>a�Ϭ(�>�������:              �?        �Ǘ'�       S�	N��-
��A�*�
u
generator_loss_1*a	   �1]�?   �1]�?      �?!   �1]�?)@xg�(�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��N�>   ��N�>      �?!   ��N�>)@�ê���=2�uE����>�f����>�������:              �?        G���       S�	�l3
��A�*�
u
generator_loss_1*a	   �P�?   �P�?      �?!   �P�?)@̨�t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �>2��Zr[v�>O�ʗ��>�������:              �?        ��g��       S�	�T8
��A�*�
u
generator_loss_1*a	   @2m�?   @2m�?      �?!   @2m�?) �E�M�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�?   @�?      �?!   @�?) q�XSu>2�FF�G ?��[�?�������:              �?        ���V�       S�	z��=
��A�*�
u
generator_loss_1*a	   ��{�?   ��{�?      �?!   ��{�?)@�efZu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)� <�vr�=2��(���>a�Ϭ(�>�������:              �?        �uP�       S�	i��B
��A�*�
u
generator_loss_1*a	   �!��?   �!��?      �?!   �!��?)@��M�)t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) i�@M_�=2�uE����>�f����>�������:              �?        ��\,�       S�	��G
��A�*�
u
generator_loss_1*a	    Ā�?    Ā�?      �?!    Ā�?)  a���w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)���ґ�=2��(���>a�Ϭ(�>�������:              �?        ��T��       S�	Q�8M
��A�*�
u
generator_loss_1*a	   �-P�?   �-P�?      �?!   �-P�?) ��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) "�D>��=2��(���>a�Ϭ(�>�������:              �?        L	~��       S�	��R
��A�*�
u
generator_loss_1*a	   `;�?   `;�?      �?!   `;�?)@Υ#E�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �0�=2�uE����>�f����>�������:              �?        �Do��       S�	���W
��A�*�
u
generator_loss_1*a	   `���?   `���?      �?!   `���?)@�Qxu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    �>    �>      �?!    �>)@�S��=2�f����>��(���>�������:              �?        >���       �{�	h ]
��A*�
u
generator_loss_1*a	    筲?    筲?      �?!    筲?) g��u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @$��>   @$��>      �?!   @$��>)��A$Y�=2�ѩ�-�>���%�>�������:              �?        ���R�       ۞��	�_b
��A(*�
u
generator_loss_1*a	   `	.�?   `	.�?      �?!   `	.�?)@~��t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@��`6�=2�uE����>�f����>�������:              �?        	�m1�       ۞��	�i�g
��AP*�
u
generator_loss_1*a	    �<�?    �<�?      �?!    �<�?)@d!q�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  �6���=2
�/eq
�>;�"�q�>�������:              �?        VBR��       ۞��	ԭ�l
��Ax*�
u
generator_loss_1*a	   �_E�?   �_E�?      �?!   �_E�?)@���C�t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �>x�>   �>x�>      �?!   �>x�>) 6���=2�uE����>�f����>�������:              �?        �@���       S�	��`r
��A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?) �˱� t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���	?   ���	?      �?!   ���	?) e���$>2����?f�ʜ�7
?�������:              �?        W'"��       S�	�%�w
��A�*�
u
generator_loss_1*a	   �+c�?   �+c�?      �?!   �+c�?)@Pl�#�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �u��>   �u��>      �?!   �u��>) r�\�>2>�?�s��>�FF�G ?�������:              �?        O�m�       S�	��-}
��A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) p'�4s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    T��>    T��>      �?!    T��>)@P^���=2pz�w�7�>I��P=�>�������:              �?        n��W�       S�	�:��
��A�*�
u
generator_loss_1*a	    �O�?    �O�?      �?!    �O�?)  �}�t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    |�>    |�>      �?!    |�>)  �I��=2a�Ϭ(�>8K�ߝ�>�������:              �?        �#P��       S�	i��
��A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) ��}ݭs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �;��>   �;��>      �?!   �;��>) D�/�/�=2})�l a�>pz�w�7�>�������:              �?        �]��       S�	<9��
��A�*�
u
generator_loss_1*a	   `�?   `�?      �?!   `�?)@�3�Mr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �6�>    �6�>      �?!    �6�>)@�G@ ��=2�ߊ4F��>})�l a�>�������:              �?        |��       S�	�3�
��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �3�+{s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��% ?   ��% ?      �?!   ��% ?) �`/SL>2>�?�s��>�FF�G ?�������:              �?        ��{&�       S�	)q�
��A�*�
u
generator_loss_1*a	   �b�?   �b�?      �?!   �b�?) �㋨�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @j�>   @j�>      �?!   @j�>)��&����=2a�Ϭ(�>8K�ߝ�>�������:              �?        L����       S�	\��
��A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) �A�`�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �*�>    �*�>      �?!    �*�>)  �h��=2E��a�W�>�ѩ�-�>�������:              �?        ).�       S�	Wd�
��A�*�
u
generator_loss_1*a	   @�S�?   @�S�?      �?!   @�S�?) ���Xw?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �&��>   �&��>      �?!   �&��>) R6`�I>2I��P=�>��Zr[v�>�������:              �?        ~�#�       S�	���
��A�*�
u
generator_loss_1*a	   `�ӱ?   `�ӱ?      �?!   `�ӱ?)@:�0��s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �'�?   �'�?      �?!   �'�?) �q��>21��a˲?6�]��?�������:              �?        ��N�       S�	n�p�
��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �k�ϱu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �T�>   �T�>      �?!   �T�>)�lqs��=28K�ߝ�>�h���`�>�������:              �?        in���       S�	7�
��A�*�
u
generator_loss_1*a	   �;�?   �;�?      �?!   �;�?)@^�Y�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    I�?    I�?      �?!    I�?) ���f>21��a˲?6�]��?�������:              �?        iPI��       S�	���
��A�*�
u
generator_loss_1*a	   �֎�?   �֎�?      �?!   �֎�?) ����Ds?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �wRޟ)>2f�ʜ�7
?>h�'�?�������:              �?        H'�       S�	eK�
��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@ L�	r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @G?   @G?      �?!   @G?)�|��e�*>2>h�'�?x?�x�?�������:              �?        �#m�       S�	t��
��A�*�
u
generator_loss_1*a	    �O�?    �O�?      �?!    �O�?)@7��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) @ ,��>21��a˲?6�]��?�������:              �?        ǿ�u�       S�	��@�
��A�*�
u
generator_loss_1*a	   �ǣ�?   �ǣ�?      �?!   �ǣ�?) ��Z�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��@�>   ��@�>      �?!   ��@�>) ���=2a�Ϭ(�>8K�ߝ�>�������:              �?        @
z.�       S�	<��
��A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) �em(�v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    �t�>    �t�>      �?!    �t�>) �	3N�=2E��a�W�>�ѩ�-�>�������:              �?        ��o�       �{�	�P]�
��A*�
u
generator_loss_1*a	   @g��?   @g��?      �?!   @g��?) I25+�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �q�>   �q�>      �?!   �q�>) :�)��=2�h���`�>�ߊ4F��>�������:              �?        7�g
�       ۞��	O+�
��A(*�
u
generator_loss_1*a	   ��J�?   ��J�?      �?!   ��J�?) D�~+�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) 1�P)�=2�f����>��(���>�������:              �?        <h��       ۞��	���
��AP*�
u
generator_loss_1*a	   ��T�?   ��T�?      �?!   ��T�?)@�Sr[w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    �m�>    �m�>      �?!    �m�>) ����=2�uE����>�f����>�������:              �?        ;~
�       ۞��	x]�
��Ax*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) q	�wt?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `I'�>   `I'�>      �?!   `I'�>)@~�b�=2��~]�[�>��>M|K�>�������:              �?        �^Q�       S�	GE	�
��A�*�
u
generator_loss_1*a	   @[�?   @[�?      �?!   @[�?) )��u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �C?   �C?      �?!   �C?) �,,>21��a˲?6�]��?�������:              �?        �?t=�       S�	�	��
��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) @��68r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @R��>   @R��>      �?!   @R��>) �&j�m�=2�uE����>�f����>�������:              �?        a�F�       S�	�n�
��A�*�
u
generator_loss_1*a	   �CW�?   �CW�?      �?!   �CW�?)@�C�4u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �6s�>   �6s�>      �?!   �6s�>)����p�>2I��P=�>��Zr[v�>�������:              �?        ����       S�	�%/�
��A�*�
u
generator_loss_1*a	   �'��?   �'��?      �?!   �'��?) �B���u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �Sk�>   �Sk�>      �?!   �Sk�>) �Ao1�=2a�Ϭ(�>8K�ߝ�>�������:              �?        �ޔ��       S�	�U���A�*�
u
generator_loss_1*a	    Ҳ?    Ҳ?      �?!    Ҳ?) @�ӊ#v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @v\�>   @v\�>      �?!   @v\�>) �I�[@�=2��>M|K�>�_�T�l�>�������:              �?        mZ	E�       S�	����A�*�
u
generator_loss_1*a	   `�Z�?   `�Z�?      �?!   `�Z�?)@.�hQu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ͺ�`I�=2�iD*L��>E��a�W�>�������:              �?        ���<�       S�	�f��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@��� �s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) i�_ہ�=2���%�>�uE����>�������:              �?        ��?�       S�	��&��A�*�
u
generator_loss_1*a	   �O��?   �O��?      �?!   �O��?) KO�ss?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ���=2��~]�[�>��>M|K�>�������:              �?        +_��       S�	�>
��A�*�
u
generator_loss_1*a	   @j��?   @j��?      �?!   @j��?) �ᔈ�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @%p�>   @%p�>      �?!   @%p�>)�\���=2E��a�W�>�ѩ�-�>�������:              �?        QEa�       S�	�b���A�*�
u
generator_loss_1*a	   �U<�?   �U<�?      �?!   �U<�?) ��t��t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ���N�=2
�/eq
�>;�"�q�>�������:              �?        ���       S�	H��%��A�*�
u
generator_loss_1*a	   ��g�?   ��g�?      �?!   ��g�?) �)U�w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    &:�>    &:�>      �?!    &:�>) la<E�=25�"�g��>G&�$�>�������:              �?        ��L��       S�	%]r+��A�*�
u
generator_loss_1*a	   ��L�?   ��L�?      �?!   ��L�?) ��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `�-?   `�-?      �?!   `�-?)@D�Z�>2��[�?1��a˲?�������:              �?        �ALq�       S�	�+A1��A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?)@�=S�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    Rh�>    Rh�>      �?!    Rh�>)  ��r�>2>�?�s��>�FF�G ?�������:              �?        �޹��       S�	�#7��A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) y��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �p�>    �p�>      �?!    �p�>) ��A�9�=2a�Ϭ(�>8K�ߝ�>�������:              �?        '����       S�	��=��A�*�
u
generator_loss_1*a	    D��?    D��?      �?!    D��?)  !4�1s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@r�pf�=2�ߊ4F��>})�l a�>�������:              �?        t. ��       S�	�$�B��A�*�
u
generator_loss_1*a	   �C��?   �C��?      �?!   �C��?)@ұY�~s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �K4@?   �K4@?      �?!   �K4@?)@З�Bi�>2���#@?�!�A?�������:              �?        Ğ��       S�	�\�H��A�*�
u
generator_loss_1*a	   @�%�?   @�%�?      �?!   @�%�?) ѻ��`r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�E?   @�E?      �?!   @�E?) )z��>21��a˲?6�]��?�������:              �?        ��7��       S�	h��N��A�*�
u
generator_loss_1*a	   �.�?   �.�?      �?!   �.�?) �p�5Uv?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �f f�=28K�ߝ�>�h���`�>�������:              �?        �����       �{�	�J�T��A*�
u
generator_loss_1*a	    ks�?    ks�?      �?!    ks�?)@<���p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @c�>   @c�>      �?!   @c�>) 	�:i��=2�h���`�>�ߊ4F��>�������:              �?        |+��       ۞��	)�Z��A(*�
u
generator_loss_1*a	   `ޯ�?   `ޯ�?      �?!   `ޯ�?)@���E�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `'��>   `'��>      �?!   `'��>) s+�	>2O�ʗ��>>�?�s��>�������:              �?        �Hl�       ۞��	�}`��AP*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) q3�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)������=2a�Ϭ(�>8K�ߝ�>�������:              �?        )�$/�       ۞��	Ҍvf��Ax*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@>�xZs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  ����=2�f����>��(���>�������:              �?        ����       S�	 wl��A�*�
u
generator_loss_1*a	   �ծ�?   �ծ�?      �?!   �ծ�?)@:|e��s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �Z8�>   �Z8�>      �?!   �Z8�>)@Na���=2pz�w�7�>I��P=�>�������:              �?        "���       S�	��tr��A�*�
u
generator_loss_1*a	    � �?    � �?      �?!    � �?) �#'��v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ss�>    ss�>      �?!    ss�>) �Z��#�=2pz�w�7�>I��P=�>�������:              �?        f�W��       S�	�1�x��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@\�M�Lr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �wh�>   �wh�>      �?!   �wh�>) �����=2pz�w�7�>I��P=�>�������:              �?        9g�H�       S�	�$�~��A�*�
u
generator_loss_1*a	    Iڰ?    Iڰ?      �?!    Iڰ?) �4�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��d�>   ��d�>      �?!   ��d�>) Wҙy��=2��(���>a�Ϭ(�>�������:              �?        arT��       S�	Y ����A�*�
u
generator_loss_1*a	   ��4�?   ��4�?      �?!   ��4�?) �F-�ip?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @�x�>   @�x�>      �?!   @�x�>) 9�	;��=2K+�E���>jqs&\��>�������:              �?        ���G�       S�	�3����A�*�
u
generator_loss_1*a	    М�?    М�?      �?!    М�?)@@-զu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)���a[s >2I��P=�>��Zr[v�>�������:              �?        �U���       S�	i"����A�*�
u
generator_loss_1*a	   @�P�?   @�P�?      �?!   @�P�?) ���r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��
?   ��
?      �?!   ��
?) �/'2>2x?�x�?��d�r?�������:              �?        `b}��       S�	ܠΖ��A�*�
u
generator_loss_1*a	   �Gα?   �Gα?      �?!   �Gα?) ����s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��m�>   ��m�>      �?!   ��m�>) �����=2pz�w�7�>I��P=�>�������:              �?        ���y�       S�	�����A�*�
u
generator_loss_1*a	   @Iа?   @Iа?      �?!   @Iа?) Y��)�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���*?   ���*?      �?!   ���*?)���<If>2I�I�)�(?�7Kaa+?�������:              �?        ���       S�	n����A�*�
u
generator_loss_1*a	   �Yر?   �Yر?      �?!   �Yر?) q71,�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    R��>    R��>      �?!    R��>)@�ɡ�-�=2pz�w�7�>I��P=�>�������:              �?        ����       S�	�*���A�*�
u
generator_loss_1*a	    w��?    w��?      �?!    w��?) �)�Qq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �}^�>   �}^�>      �?!   �}^�>) dx�g��=2pz�w�7�>I��P=�>�������:              �?        ;?�A�       S�	��S���A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) ��g�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)��lK�#>2����?f�ʜ�7
?�������:              �?        �<��       S�	�v���A�*�
u
generator_loss_1*a	   @=��?   @=��?      �?!   @=��?)�<}�{�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @P��>   @P��>      �?!   @P��>)�@� �>2>�?�s��>�FF�G ?�������:              �?        gjd��       S�	����A�*�
u
generator_loss_1*a	   @?   @?      �?!   @?) ��S�,s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    /��>    /��>      �?!    /��>) ��U��=2I��P=�>��Zr[v�>�������:              �?        ��7�       S�	�b����A�*�
u
generator_loss_1*a	    �m�?    �m�?      �?!    �m�?)@TD:u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    ͸�>    ͸�>      �?!    ͸�>) �ث~�=2E��a�W�>�ѩ�-�>�������:              �?        �	�       S�	]����A�*�
u
generator_loss_1*a	   `�t�?   `�t�?      �?!   `�t�?)@�V��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) �BGh~>2��[�?1��a˲?�������:              �?        eV���       S�	զH���A�*�
u
generator_loss_1*a	   @fY�?   @fY�?      �?!   @fY�?) q�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �VZ�>   �VZ�>      �?!   �VZ�>) ��!�=2�f����>��(���>�������:              �?        *ݔ�       S�	ۅ����A�*�
u
generator_loss_1*a	   `���?   `���?      �?!   `���?)@b*r�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �(��>   �(��>      �?!   �(��>) ���O��=2�f����>��(���>�������:              �?        ����       �{�	Jޮ���A*�
u
generator_loss_1*a	   `X�?   `X�?      �?!   `X�?)@>H/u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `K2�>   `K2�>      �?!   `K2�>) �=�R��=2�iD*L��>E��a�W�>�������:              �?        O䓝�       ۞��	F�����A(*�
u
generator_loss_1*a	   �-�?   �-�?      �?!   �-�?)@�㮊
t?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@f7���>21��a˲?6�]��?�������:              �?        f�}��       ۞��	�
@���AP*�
u
generator_loss_1*a	   `s3�?   `s3�?      �?!   `s3�?)@��2�gp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `f�?   `f�?      �?!   `f�?)@
S�>26�]��?����?�������:              �?        �΋!�       ۞��	+����Ax*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@���-s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)��w��8 >26�]��?����?�������:              �?        �@:��       S�	�z����A�*�
u
generator_loss_1*a	   ��z�?   ��z�?      �?!   ��z�?) �A�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @l:�>   @l:�>      �?!   @l:�>)�0�Ef� >2I��P=�>��Zr[v�>�������:              �?        ��~1�       S�	��4���A�*�
u
generator_loss_1*a	   `���?   `���?      �?!   `���?)@"k�^Hs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `zE�>   `zE�>      �?!   `zE�>) ���3��=2�ѩ�-�>���%�>�������:              �?        ��L2�       S�	98� ��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@���s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `�<�>   `�<�>      �?!   `�<�>)@B��f��=2���%�>�uE����>�������:              �?        �0�T�       S�	B����A�*�
u
generator_loss_1*a	   ��Y�?   ��Y�?      �?!   ��Y�?)@�r�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    �*�>    �*�>      �?!    �*�>) Y��=2�iD*L��>E��a�W�>�������:              �?        �~��       S�	�N��A�*�
u
generator_loss_1*a	   �ݰ?   �ݰ?      �?!   �ݰ?) $�s�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �''�>   �''�>      �?!   �''�>)@"��'��=2�f����>��(���>�������:              �?        xOl�       S�	T����A�*�
u
generator_loss_1*a	   �	�?   �	�?      �?!   �	�?)@L��gp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    W�>    W�>      �?!    W�>)  &ԁ��=2��(���>a�Ϭ(�>�������:              �?        p��       S�	&�)��A�*�
u
generator_loss_1*a	   `[��?   `[��?      �?!   `[��?)@��c�Rs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @`��>   @`��>      �?!   @`��>) 3�e��=2�ߊ4F��>})�l a�>�������:              �?        ��aN�       S�	�.� ��A�*�
u
generator_loss_1*a	   ��E�?   ��E�?      �?!   ��E�?) ��	�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �,��>   �,��>      �?!   �,��>) �u���>2��Zr[v�>O�ʗ��>�������:              �?        ��5z�       S�	�'��A�*�
u
generator_loss_1*a	   �ɖ�?   �ɖ�?      �?!   �ɖ�?) ���Us?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �o�>    �o�>      �?!    �o�>)@X`!�=�=2���%�>�uE����>�������:              �?        �Zi�       S�	�=m-��A�*�
u
generator_loss_1*a	   `�/�?   `�/�?      �?!   `�/�?)@
i�Xvr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  YȬ��=2jqs&\��>��~]�[�>�������:              �?        �����       S�	��3��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) @��7r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) ��C�F>2��ڋ?�.�?�������:              �?        ��L��       S�	'c:��A�*�
u
generator_loss_1*a	   @�Ǳ?   @�Ǳ?      �?!   @�Ǳ?) ����s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @j��>   @j��>      �?!   @j��>)�Ȥ��d>2��Zr[v�>O�ʗ��>�������:              �?        oT���       S�	���@��A�*�
u
generator_loss_1*a	   �zŰ?   �zŰ?      �?!   �zŰ?) �e�z�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �a��>   �a��>      �?!   �a��>) $�{���=2�ߊ4F��>})�l a�>�������:              �?        �P{�       S�	@�hG��A�*�
u
generator_loss_1*a	   ��i�?   ��i�?      �?!   ��i�?)@�ʑ��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �b�?   �b�?      �?!   �b�?)���2CC*>2>h�'�?x?�x�?�������:              �?        j����       S�	���M��A�*�
u
generator_loss_1*a	   @㽰?   @㽰?      �?!   @㽰?) �c��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    "��>    "��>      �?!    "��>) d�����=2��(���>a�Ϭ(�>�������:              �?        3{��       S�	�<fT��A�*�
u
generator_loss_1*a	   @8��?   @8��?      �?!   @8��?) �5�'Cs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @h�>   @h�>      �?!   @h�>) 	4��-�=2})�l a�>pz�w�7�>�������:              �?        >� ��       S�	i>�Z��A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) �bV�bs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �s��>   �s��>      �?!   �s��>) ���=�=28K�ߝ�>�h���`�>�������:              �?        �j��       S�	pBxa��A�*�
u
generator_loss_1*a	   �Ry�?   �Ry�?      �?!   �Ry�?)@�^3=�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) ia���=2���%�>�uE����>�������:              �?        h�"�       �{�	��g��A*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@���2�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@�{����=2�uE����>�f����>�������:              �?        )� 1�       ۞��	�n��A(*�
u
generator_loss_1*a	    n��?    n��?      �?!    n��?) @���%q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ���Ck�=2��~]�[�>��>M|K�>�������:              �?        TQC�       ۞��	I~u��AP*�
u
generator_loss_1*a	   �Wf�?   �Wf�?      �?!   �Wf�?) �~}�(u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �Bݟ�=2E��a�W�>�ѩ�-�>�������:              �?        ��K�       ۞��	�i�{��Ax*�
u
generator_loss_1*a	   �m�?   �m�?      �?!   �m�?)@ɤ�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    N�>    N�>      �?!    N�>) �0�u�=2�_�T�l�>�iD*L��>�������:              �?        �+���       S�	Z�p���A�*�
u
generator_loss_1*a	   �Q��?   �Q��?      �?!   �Q��?)@j��@q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �UA�&�=28K�ߝ�>�h���`�>�������:              �?        ��4��       S�	VG���A�*�
u
generator_loss_1*a	   ��L�?   ��L�?      �?!   ��L�?) �4. �r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @m��>   @m��>      �?!   @m��>)�����i�=2�ѩ�-�>���%�>�������:              �?        U]���       S�	以���A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@01Q�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @͙�>   @͙�>      �?!   @͙�>) ����=2��~]�[�>��>M|K�>�������:              �?        ��A��       S�	X(N���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) $uc3"r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@6I2���=2��~]�[�>��>M|K�>�������:              �?        ����       S�	������A�*�
u
generator_loss_1*a	   @c��?   @c��?      �?!   @c��?) �W���s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    E��>    E��>      �?!    E��>) R�Y�p�=2�iD*L��>E��a�W�>�������:              �?        �(��       S�	ݮ����A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@l�;D�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �s�>   �s�>      �?!   �s�>) ���3��=2�����>
�/eq
�>�������:              �?        R����       S�	Oj���A�*�
u
generator_loss_1*a	   �(b�?   �(b�?      �?!   �(b�?) ɗ���p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �;�>    �;�>      �?!    �;�>)  ��U��=2�ߊ4F��>})�l a�>�������:              �?        pc:�       S�	AA'���A�*�
u
generator_loss_1*a	    6��?    6��?      �?!    6��?)@�Sq�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @;��>   @;��>      �?!   @;��>) i;t��=2��>M|K�>�_�T�l�>�������:              �?        mQX�       S�	C����A�*�
u
generator_loss_1*a	    X��?    X��?      �?!    X��?)  �p�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    i�>    i�>      �?!    i�>) n��\�=2['�?��>K+�E���>�������:              �?        ��fB�       S�	�����A�*�
u
generator_loss_1*a	   �Ӆ�?   �Ӆ�?      �?!   �Ӆ�?)@�a��0s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) t�#��=2['�?��>K+�E���>�������:              �?        r�z��       S�	!y���A�*�
u
generator_loss_1*a	   �#�?   �#�?      �?!   �#�?) ��t?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �/��>   �/��>      �?!   �/��>) ���K��=2;�"�q�>['�?��>�������:              �?        �8|U�       S�	A;���A�*�
u
generator_loss_1*a	   �䮰?   �䮰?      �?!   �䮰?)@��~Aeq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �a�>    �a�>      �?!    �a�>) �uA��=2
�/eq
�>;�"�q�>�������:              �?        C�i�       S�	�V���A�*�
u
generator_loss_1*a	   �[�?   �[�?      �?!   �[�?) !��L*r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �r�>    �r�>      �?!    �r�>)  (��.�=2�����>
�/eq
�>�������:              �?        �7��       S�	N�����A�*�
u
generator_loss_1*a	   ��)�?   ��)�?      �?!   ��)�?) d��Lir?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `׶�>   `׶�>      �?!   `׶�>) ��=ũ�=2
�/eq
�>;�"�q�>�������:              �?        o�1��       S�	̑����A�*�
u
generator_loss_1*a	   �[��?   �[��?      �?!   �[��?)@���&s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    k��>    k��>      �?!    k��>) �9���=2��~���>�XQ��>�������:              �?        �
��       S�	H;|���A�*�
u
generator_loss_1*a	   @_d�?   @_d�?      �?!   @_d�?) 	���r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�I�>   @�I�>      �?!   @�I�>) �S��S�=2�XQ��>�����>�������:              �?        �x���       S�	��Z���A�*�
u
generator_loss_1*a	   �=�?   �=�?      �?!   �=�?) y�|��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �E�>    �E�>      �?!    �E�>)  �ߑ�=2�iD*L��>E��a�W�>�������:              �?        ~P|�       S�	�TC���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) anmIs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �2��>   �2��>      �?!   �2��>)@.��Y�=2jqs&\��>��~]�[�>�������:              �?        �t.��       �{�	]$���A*�
u
generator_loss_1*a	   @y��?   @y��?      �?!   @y��?) ���{bs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    @3�>    @3�>      �?!    @3�>)@ �},�=2�f����>��(���>�������:              �?        Z�0�       ۞��	Z���A(*�
u
generator_loss_1*a	   �	O�?   �	O�?      �?!   �	O�?)@�1��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@n��|�=2��~]�[�>��>M|K�>�������:              �?        �� ��       ۞��	��	��AP*�
u
generator_loss_1*a	   �f�?   �f�?      �?!   �f�?) �H���r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �%�>   �%�>      �?!   �%�>) ��x�=2��~���>�XQ��>�������:              �?        �I��       ۞��	۾���Ax*�
u
generator_loss_1*a	    *ذ?    *ذ?      �?!    *ذ?)@�α��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `W��>   `W��>      �?!   `W��>) ����K�=2['�?��>K+�E���>�������:              �?        �l�       S�	қ���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@�0�(p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) .򂄋�=2['�?��>K+�E���>�������:              �?        ����       S�	����A�*�
u
generator_loss_1*a	    ]|�?    ]|�?      �?!    ]|�?)@���p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�p�>   @�p�>      �?!   @�p�>) )eP��=2��~���>�XQ��>�������:              �?        ����       S�	���%��A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?)@�;uBt?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) g����=25�"�g��>G&�$�>�������:              �?        ;��n�       S�	ҫ�,��A�*�
u
generator_loss_1*a	    �2�?    �2�?      �?!    �2�?) �� P|r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�h��s��=2G&�$�>�*��ڽ>�������:              �?        -�ol�       S�	�#�3��A�*�
u
generator_loss_1*a	    {��?    {��?      �?!    {��?) �qT�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @J�>   @J�>      �?!   @J�>) ٷ!SAw=2��n����>�u`P+d�>�������:              �?        ^���       S�	��:��A�*�
u
generator_loss_1*a	   �]�?   �]�?      �?!   �]�?) Q���2r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)��B�	�=2��>M|K�>�_�T�l�>�������:              �?        ����       S�	���A��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@�*t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @�K�>   @�K�>      �?!   @�K�>) $�X�=2�XQ��>�����>�������:              �?        BZ�$�       S�	��H��A�*�
u
generator_loss_1*a	   � W�?   � W�?      �?!   � W�?)@�j��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @E`�>   @E`�>      �?!   @E`�>)��x�Ռ=2�*��ڽ>�[�=�k�>�������:              �?        �f�       S�	��O��A�*�
u
generator_loss_1*a	   @{��?   @{��?      �?!   @{��?) i��9`q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�S�>   @�S�>      �?!   @�S�>) Q׏�Ò=2�[�=�k�>��~���>�������:              �?        ��3�       S�	h�V��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) i 	tJt?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @xg�>   @xg�>      �?!   @xg�>)����`6�=2G&�$�>�*��ڽ>�������:              �?        �I�       S�	��^��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@���q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ��R��=25�"�g��>G&�$�>�������:              �?        ��1��       S�	X%e��A�*�
u
generator_loss_1*a	   `ሱ?   `ሱ?      �?!   `ሱ?)@���q7s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�y�>   @�y�>      �?!   @�y�>) Y����=2�[�=�k�>��~���>�������:              �?        ԙ&L�       S�	�Ml��A�*�
u
generator_loss_1*a	   �`J�?   �`J�?      �?!   �`J�?) ��f�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �g�>    �g�>      �?!    �g�>)@��S�ѐ=2�*��ڽ>�[�=�k�>�������:              �?        E4��       S�	�Xss��A�*�
u
generator_loss_1*a	    �;�?    �;�?      �?!    �;�?)  !�Ώr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) )���1�=2�*��ڽ>�[�=�k�>�������:              �?        �6��       S�	ߺ�z��A�*�
u
generator_loss_1*a	   @-8�?   @-8�?      �?!   @-8�?) �?+�t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �Z�>   �Z�>      �?!   �Z�>) ��3T~=2�u`P+d�>0�6�/n�>�������:              �?        �,\�       S�	������A�*�
u
generator_loss_1*a	   ��ױ?   ��ױ?      �?!   ��ױ?) �w�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)���zi�b=2�5�L�>;9��R�>�������:              �?        +�N��       S�	����A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@�$�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �yت>   �yت>      �?!   �yت>) E�~�f=2;9��R�>���?�ګ>�������:              �?         �~��       S�	\Y���A�*�
u
generator_loss_1*a	   ��D�?   ��D�?      �?!   ��D�?)@��£r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �h˺>   �h˺>      �?!   �h˺>) B-ǖo�=25�"�g��>G&�$�>�������:              �?        mm���       �{�	>�0���A*�
u
generator_loss_1*a	    d��?    d��?      �?!    d��?)@�r�щq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �J��>   �J��>      �?!   �J��>) 2m�JQf=2;9��R�>���?�ګ>�������:              �?        u+�d�       ۞��	��r���A(*�
u
generator_loss_1*a	   `e��?   `e��?      �?!   `e��?)@N�Kc4q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��0�>   ��0�>      �?!   ��0�>)@255�bp=2����>豪}0ڰ>�������:              �?        ���3�       ۞��	k�����AP*�
u
generator_loss_1*a	   @�E�?   @�E�?      �?!   @�E�?) �j�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    z8�>    z8�>      �?!    z8�>)  ����j=2���?�ګ>����>�������:              �?        	mn��       ۞��	�����Ax*�
u
generator_loss_1*a	   ��˲?   ��˲?      �?!   ��˲?)@>�Wv?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��ȣ>   ��ȣ>      �?!   ��ȣ>)@����vX=2��|�~�>���]���>�������:              �?        ���       S�	�OH���A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) ��v!p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `A��>   `A��>      �?!   `A��>)@�7X=2��|�~�>���]���>�������:              �?        Q�d�       S�	K����A�*�
u
generator_loss_1*a	    h��?    h��?      �?!    h��?)  �2s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)��n,Ba=2�5�L�>;9��R�>�������:              �?        ��a��       S�	/�����A�*�
u
generator_loss_1*a	    ]A�?    ]A�?      �?!    ]A�?) H���l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `�G�>   `�G�>      �?!   `�G�>) �q8ʪ=2['�?��>K+�E���>�������:              �?        ��fJ�       S�	�:!���A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@���?t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   @�5�>   @�5�>      �?!   @�5�>)�pX�x�=25�"�g��>G&�$�>�������:              �?        ��F��       S�	 ����A�*�
u
generator_loss_1*a	   @�װ?   @�װ?      �?!   @�װ?) ��ߺq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���}>   ���}>      �?!   ���}>)�|y�=2f^��`{>�����~>�������:              �?        4yr#�       S�	M%����A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?) dVr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �&�>   �&�>      �?!   �&�>)@t�`�=2[#=�؏�>K���7�>�������:              �?        ބ�N�       S�	Kd-���A�*�
u
generator_loss_1*a	   �(�?   �(�?      �?!   �(�?)@4�6�fr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �D�>   �D�>      �?!   �D�>)�\�ew� =2T�L<�>��z!�?�>�������:              �?        ��$��       S�	�����A�*�
u
generator_loss_1*a	    !b�?    !b�?      �?!    !b�?)@�:��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@�_c4=2�4[_>��>
�}���>�������:              �?        lS6��       S�	1�����A�*�
u
generator_loss_1*a	   �_�?   �_�?      �?!   �_�?) �O�D�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    u͖>    u͖>      �?!    u͖>) Ȼ�?@=2X$�z�>.��fc��>�������:              �?        ��3�       S�	V�O���A�*�
u
generator_loss_1*a	   �nx�?   �nx�?      �?!   �nx�?)@T vs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    R!�>    R!�>      �?!    R!�>)  �2b=2�5�L�>;9��R�>�������:              �?        ��`�       S�	P÷���A�*�
u
generator_loss_1*a	   �O/�?   �O/�?      �?!   �O/�?)@�Ңur?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @k��>   @k��>      �?!   @k��>)�t綾��=2['�?��>K+�E���>�������:              �?        ��Z��       S�	��#��A�*�
u
generator_loss_1*a	    �P�?    �P�?      �?!    �P�?)@t�@Ȣp?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�b�>   @�b�>      �?!   @�b�>)�`=����=2a�Ϭ(�>8K�ߝ�>�������:              �?        �j��       S�	Gy���A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) {xq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@B�=��=2K+�E���>jqs&\��>�������:              �?        5V+��       S�	 ���A�*�
u
generator_loss_1*a	   @�?   @�?      �?!   @�?) )j��s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    C��>    C��>      �?!    C��>) ���M=2�XQ��>�����>�������:              �?        g���       S�	��u��A�*�
u
generator_loss_1*a	   `c-�?   `c-�?      �?!   `c-�?)@6e��pr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ru�>    ru�>      �?!    ru�>)  6����=2E��a�W�>�ѩ�-�>�������:              �?        +K�V�       S�	�
#��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) ����Oq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @#��>   @#��>      �?!   @#��>) ��.��=2�f����>��(���>�������:              �?        sxZz�       S�	��*��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@\���Tq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) �q�"�=2��~]�[�>��>M|K�>�������:              �?        ��i�       S�	�V2��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) @�Tq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@ �q`#�=2�XQ��>�����>�������:              �?        �5�a�       �{�	ބ�9��A*�
u
generator_loss_1*a	   �뱰?   �뱰?      �?!   �뱰?)@PQ@�kq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `e��>   `e��>      �?!   `e��>)@N�P 2�=2���%�>�uE����>�������:              �?        =X,��       ۞��	�J!A��A(*�
u
generator_loss_1*a	   ��0�?   ��0�?      �?!   ��0�?) $��bp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @m��>   @m��>      �?!   @m��>)��(���=2['�?��>K+�E���>�������:              �?        ����       ۞��	�Q�H��AP*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?)@v's>s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��Z�>   ��Z�>      �?!   ��Z�>) DL�;�=2I��P=�>��Zr[v�>�������:              �?        ��s��       ۞��	��RP��Ax*�
u
generator_loss_1*a	   ��ְ?   ��ְ?      �?!   ��ְ?) !���q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@L@�=2��~���>�XQ��>�������:              �?        ���|�       S�	%��W��A�*�
u
generator_loss_1*a	    �(�?    �(�?      �?!    �(�?)@\rmgr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `��?   `��?      �?!   `��?)@F��S�>2�FF�G ?��[�?�������:              �?        �<���       S�	1��_��A�*�
u
generator_loss_1*a	   ��D�?   ��D�?      �?!   ��D�?)@�dL�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ߿�>    ߿�>      �?!    ߿�>)@�C���=2})�l a�>pz�w�7�>�������:              �?        ���F�       S�	��g��A�*�
u
generator_loss_1*a	   @Z��?   @Z��?      �?!   @Z��?) �LOq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ���*�=2})�l a�>pz�w�7�>�������:              �?        �ا��       S�	m��n��A�*�
u
generator_loss_1*a	   �磰?   �磰?      �?!   �磰?) ��	^Nq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @I�>   @I�>      �?!   @I�>) Yo��m�=2�ߊ4F��>})�l a�>�������:              �?        �����       S�	��v��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@�+<�t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@��KA�=2})�l a�>pz�w�7�>�������:              �?        54��       S�	,4G~��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@�^4/p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) d"�
�>21��a˲?6�]��?�������:              �?        [Tu7�       S�	"$����A�*�
u
generator_loss_1*a	   @�Y�?   @�Y�?      �?!   @�Y�?) �_��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    l�?    l�?      �?!    l�?)  Y�\�>2��[�?1��a˲?�������:              �?        U=�{�       S�	�����A�*�
u
generator_loss_1*a	   �X�?   �X�?      �?!   �X�?)@�IG��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��/�>   ��/�>      �?!   ��/�>) �Z�w�=2})�l a�>pz�w�7�>�������:              �?        �8GK�       S�	:�H���A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@`Ŕr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) "̡�D >2I��P=�>��Zr[v�>�������:              �?        ����       S�	N����A�*�
u
generator_loss_1*a	    �(�?    �(�?      �?!    �(�?)@L��Qp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @�B�>   @�B�>      �?!   @�B�>)�P�����=28K�ߝ�>�h���`�>�������:              �?        g#���       S�	������A�*�
u
generator_loss_1*a	   ��1�?   ��1�?      �?!   ��1�?)@2���zr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �yB�>   �yB�>      �?!   �yB�>) R�����=2�iD*L��>E��a�W�>�������:              �?        ��r��       S�	�����A�*�
u
generator_loss_1*a	   �9~�?   �9~�?      �?!   �9~�?) q`� s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  Rz���=2�_�T�l�>�iD*L��>�������:              �?        `=�V�       S�	yC���A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) 1�G%�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @~�>   @~�>      �?!   @~�>) �Q����=2pz�w�7�>I��P=�>�������:              �?        �s8'�       S�	\�����A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) d��4&q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    Oc�>    Oc�>      �?!    Oc�>) �Y��.	>2O�ʗ��>>�?�s��>�������:              �?        /$��       S�	�;����A�*�
u
generator_loss_1*a	   �ͱ?   �ͱ?      �?!   �ͱ?)@�}�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �̵?   �̵?      �?!   �̵?) ₶.�)>2f�ʜ�7
?>h�'�?�������:              �?        ����       S�	�ŵ���A�*�
u
generator_loss_1*a	   @�q�?   @�q�?      �?!   @�q�?)��E^�k?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   �Ԍ�>   �Ԍ�>      �?!   �Ԍ�>) 0��>2��Zr[v�>O�ʗ��>�������:              �?        �K��       S�	KĢ���A�*�
u
generator_loss_1*a	    �ү?    �ү?      �?!    �ү?) �@S7�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��W�>   ��W�>      �?!   ��W�>) dX?��=2�ߊ4F��>})�l a�>�������:              �?        B�q
�       S�	��|���A�*�
u
generator_loss_1*a	   �۰?   �۰?      �?!   �۰?)@V�N��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @Nb�>   @Nb�>      �?!   @Nb�>)�X(��>2��Zr[v�>O�ʗ��>�������:              �?        ��L��       �{�	vD���A*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@p{߬Kq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)�(�?��=2�h���`�>�ߊ4F��>�������:              �?        �h��       ۞��	*; ���A(*�
u
generator_loss_1*a	   �J2�?   �J2�?      �?!   �J2�?) 9��3ep?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    Nt�?    Nt�?      �?!    Nt�?) @|l��?2>	� �?����=��?�������:              �?        j�[�       ۞��	�q����AP*�
u
generator_loss_1*a	   �l��?   �l��?      �?!   �l��?) �?��tq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@.�V���=2�ߊ4F��>})�l a�>�������:              �?        R'��       ۞��	�d����Ax*�
u
generator_loss_1*a	   @V��?   @V��?      �?!   @V��?) �8c^�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) $��=2})�l a�>pz�w�7�>�������:              �?        �i�n�       S�	�(���A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@4�i>r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    aY�>    aY�>      �?!    aY�>) �D7��=2a�Ϭ(�>8K�ߝ�>�������:              �?        6C��       S�	�g�
��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ��p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @�M?   @�M?      �?!   @�M?) I�>�?>2�T7��?�vV�R9?�������:              �?        C����       S�	����A�*�
u
generator_loss_1*a	   @�d�?   @�d�?      �?!   @�d�?) �����r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ��,g��=2E��a�W�>�ѩ�-�>�������:              �?        ���       S�	�c���A�*�
u
generator_loss_1*a	   �ڷ�?   �ڷ�?      �?!   �ڷ�?) ����s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) ��Bo>21��a˲?6�]��?�������:              �?        ^=��       S�	i�y"��A�*�
u
generator_loss_1*a	    _)�?    _)�?      �?!    _)�?) �(Sp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    \6?    \6?      �?!    \6?) 8�ŀ� >26�]��?����?�������:              �?        {>u�       S�	��t*��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@(v�m@s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �?    �?      �?!    �?)  ��p+>2>h�'�?x?�x�?�������:              �?        [ 3�       S�	P�v2��A�*�
u
generator_loss_1*a	   ��d�?   ��d�?      �?!   ��d�?)@d���r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    .m?    .m?      �?!    .m?)@����2>2x?�x�?��d�r?�������:              �?        ����       S�	�z:��A�*�
u
generator_loss_1*a	    s��?    s��?      �?!    s��?)@\����s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `¡?   `¡?      �?!   `¡?) ���s!>26�]��?����?�������:              �?        RF� �       S�	�rvB��A�*�
u
generator_loss_1*a	    ,��?    ,��?      �?!    ,��?)  y��cq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) A�o/�>2O�ʗ��>>�?�s��>�������:              �?        �g�       S�	��J��A�*�
u
generator_loss_1*a	   �.m�?   �.m�?      �?!   �.m�?)@����r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��{�>   ��{�>      �?!   ��{�>)����XZ�=28K�ߝ�>�h���`�>�������:              �?        �r�       S�	b�R��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)  i
c\s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �4�>    �4�>      �?!    �4�>)@<vg.��=2�ߊ4F��>})�l a�>�������:              �?        ��(��       S�	�u�Z��A�*�
u
generator_loss_1*a	   �(.�?   �(.�?      �?!   �(.�?)@&�}�rr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �h��>   �h��>      �?!   �h��>) �::��=2E��a�W�>�ѩ�-�>�������:              �?        ����       S�	���b��A�*�
u
generator_loss_1*a	    �v�?    �v�?      �?!    �v�?)  ���p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �<�>    �<�>      �?!    �<�>) �m�.�=2a�Ϭ(�>8K�ߝ�>�������:              �?        j�p�       S�	���j��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@��|�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  in]H�=2pz�w�7�>I��P=�>�������:              �?        5K��       S�	�L�r��A�*�
u
generator_loss_1*a	    �a�?    �a�?      �?!    �a�?)@���r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �!�>    �!�>      �?!    �!�>)@4���C�=2�h���`�>�ߊ4F��>�������:              �?        c�,�       S�	�}�z��A�*�
u
generator_loss_1*a	    J�?    J�?      �?!    J�?)@h.r"s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@.D�=2�ѩ�-�>���%�>�������:              �?        �D���       S�	a����A�*�
u
generator_loss_1*a	   `�/�?   `�/�?      �?!   `�/�?)@Z�ur?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  @�lS�=2�ߊ4F��>})�l a�>�������:              �?        �T�A�       S�	��@���A�*�
u
generator_loss_1*a	   �q�?   �q�?      �?!   �q�?) �,Jt?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �/��>   �/��>      �?!   �/��>)@�]�J��=2���%�>�uE����>�������:              �?        �SϪ�       �{�	��E���A*�
u
generator_loss_1*a	   �|а?   �|а?      �?!   �|а?) �L/��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) B�7�>2>�?�s��>�FF�G ?�������:              �?        ՚�1�       ۞��	?����A(*�
u
generator_loss_1*a	   ��z�?   ��z�?      �?!   ��z�?) ��s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) 	�P�1�=2�uE����>�f����>�������:              �?        �lo��       ۞��	+%����AP*�
u
generator_loss_1*a	   �'��?   �'��?      �?!   �'��?) ��k�Xs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �P)�>   �P)�>      �?!   �P)�>)�ķ�Xc�=2a�Ϭ(�>8K�ߝ�>�������:              �?        �W*��       ۞��	eX����Ax*�
u
generator_loss_1*a	   �Aư?   �Aư?      �?!   �Aư?) 1^�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �J�>    �J�>      �?!    �J�>) �(����=2��(���>a�Ϭ(�>�������:              �?        �)���       S�	z{3���A�*�
u
generator_loss_1*a	   ��#�?   ��#�?      �?!   ��#�?) �i�YGp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �b�>   �b�>      �?!   �b�>)������=2�iD*L��>E��a�W�>�������:              �?        ����       S�	S�~���A�*�
u
generator_loss_1*a	   �埰?   �埰?      �?!   �埰?) +s	Fq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    4G�>    4G�>      �?!    4G�>)@�E/?��=2�uE����>�f����>�������:              �?        �m�b�       S�	�_����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ���=)p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �~�>    �~�>      �?!    �~�>)  o����=2['�?��>K+�E���>�������:              �?        ����       S�	�����A�*�
u
generator_loss_1*a	   ��r�?   ��r�?      �?!   ��r�?)@�Ę1s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `Ϡ�>   `Ϡ�>      �?!   `Ϡ�>)@����G�=2K+�E���>jqs&\��>�������:              �?        E���       S�	�85���A�*�
u
generator_loss_1*a	    L��?    L��?      �?!    L��?)  iˎ>q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �)��>   �)��>      �?!   �)��>)�x�pp��=2;�"�q�>['�?��>�������:              �?        �����       S�	�����A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) AI��`q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  ��j>2��Zr[v�>O�ʗ��>�������:              �?        x}�       S�	\�����A�*�
u
generator_loss_1*a	   ��6�?   ��6�?      �?!   ��6�?)�X��Urn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �C�>    �C�>      �?!    �C�>) �U(���=2E��a�W�>�ѩ�-�>�������:              �?        ���       S�	�P*���A�*�
u
generator_loss_1*a	    v�?    v�?      �?!    v�?) �VX��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @s8�>   @s8�>      �?!   @s8�>) )����=2�uE����>�f����>�������:              �?        ��f�       S�	��{���A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) i��1Yq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ɟ�>    ɟ�>      �?!    ɟ�>) ���i�=2�ߊ4F��>})�l a�>�������:              �?        �)��       S�	�*����A�*�
u
generator_loss_1*a	   ��g�?   ��g�?      �?!   ��g�?)@B5���p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    G��>    G��>      �?!    G��>) 4���=2�f����>��(���>�������:              �?        �����       S�	g;��A�*�
u
generator_loss_1*a	   `p�?   `p�?      �?!   `p�?) ��|��o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �p��>   �p��>      �?!   �p��>) d���=2})�l a�>pz�w�7�>�������:              �?        @4���       S�	K����A�*�
u
generator_loss_1*a	   �i�?   �i�?      �?!   �i�?) LX���o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    3v�>    3v�>      �?!    3v�>) ��pu��=2I��P=�>��Zr[v�>�������:              �?        ��]��       S�	�h��A�*�
u
generator_loss_1*a	    ~��?    ~��?      �?!    ~��?)@8j�u?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �<}�>   �<}�>      �?!   �<}�>) �4b�]�=2���%�>�uE����>�������:              �?        ����       S�	��x ��A�*�
u
generator_loss_1*a	   �>c�?   �>c�?      �?!   �>c�?)@�L�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �G!�>   �G!�>      �?!   �G!�>) �9�ʛ�=2�f����>��(���>�������:              �?        �[�t�       S�	%K�(��A�*�
u
generator_loss_1*a	   �n�?   �n�?      �?!   �n�?)@hۈ@�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �!�>   �!�>      �?!   �!�>) ���SB�=2�h���`�>�ߊ4F��>�������:              �?        u&��       S�	�sR1��A�*�
u
generator_loss_1*a	   `ͅ�?   `ͅ�?      �?!   `ͅ�?)@.��&�w?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �ظ�>   �ظ�>      �?!   �ظ�>) ��YO=2K���7�>u��6
�>�������:              �?        �N��       S�	~��9��A�*�
u
generator_loss_1*a	    ˲?    ˲?      �?!    ˲?)@t��
v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ��P�=   ��P�=      �?!   ��P�=) P�<2����%�=f;H�\Q�=�������:              �?        ?����       S�	o�AB��A�*�
u
generator_loss_1*a	   �dt�?   �dt�?      �?!   �dt�?)@�+�1Iu?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   �:��=   �:��=      �?!   �:��=) �EN�_�;2��-��J�=�K���=�������:              �?        c���       �{�	֡J��A*�
u
generator_loss_1*a	   �k��?   �k��?      �?!   �k��?) DbŌ�x?2�{ �ǳ�?� l(��?�������:              �?        
w
discriminator_loss*a	   �I��=   �I��=      �?!   �I��=)@�,$��;2�K���=�9�e��=�������:              �?        �I�E�       ۞��	�7'S��A(*�
u
generator_loss_1*a	   �w��?   �w��?      �?!   �w��?) ��k/t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   ���A>   ���A>      �?!   ���A>)@����&�<2p��Dp�@>/�p`B>�������:              �?        ��Y�       ۞��	^̧[��AP*�
u
generator_loss_1*a	   ��װ?   ��װ?      �?!   ��װ?) �N��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���O>   ���O>      �?!   ���O>)�H_P�<2������M>28���FP>�������:              �?        ;�+�       ۞��	�/d��Ax*�
u
generator_loss_1*a	   �02�?   �02�?      �?!   �02�?) �D{r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �urd>   �urd>      �?!   �urd>)@���X!�<2�����0c>cR�k�e>�������:              �?        	�Z�       S�	�Ĺl��A�*�
u
generator_loss_1*a	   @�x�?   @�x�?      �?!   @�x�?) ��=�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��ͅ>   ��ͅ>      �?!   ��ͅ>)@��=2u��6
�>T�L<�>�������:              �?        �c:�       S�	dCu��A�*�
u
generator_loss_1*a	   �J��?   �J��?      �?!   �J��?)@\�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  by/�=20�6�/n�>5�"�g��>�������:              �?        D��K�       S�	�a�}��A�*�
u
generator_loss_1*a	   �=�?   �=�?      �?!   �=�?) Q��{p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��}�>   ��}�>      �?!   ��}�>) 5^N�=2�_�T�l�>�iD*L��>�������:              �?        ����       S�	�p���A�*�
u
generator_loss_1*a	    lP�?    lP�?      �?!    lP�?)@�|l�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��h0?   ��h0?      �?!   ��h0?) ��?�p>2��VlQ.?��bȬ�0?�������:              �?        �b��       S�	6����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@���=p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �S�$?   �S�$?      �?!   �S�$?)@mgk�Z>2U�4@@�$?+A�F�&?�������:              �?        �0�@�       S�	�]Η��A�*�
u
generator_loss_1*a	    ȁ�?    ȁ�?      �?!    ȁ�?)  bفo?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �"?    �"?      �?!    �"?) Iy�eT>2�S�F !?�[^:��"?�������:              �?        =R8�       S�	C�u���A�*�
u
generator_loss_1*a	   �>&�?   �>&�?      �?!   �>&�?)@��0�v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    L��>    L��>      �?!    L��>)@0N�$(�=2pz�w�7�>I��P=�>�������:              �?        c'���       S�	�L���A�*�
u
generator_loss_1*a	   �c�?   �c�?      �?!   �c�?) ��G��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �;�>   �;�>      �?!   �;�>)����>2��Zr[v�>O�ʗ��>�������:              �?        ֱ���       S�	������A�*�
u
generator_loss_1*a	    �k�?    �k�?      �?!    �k�?)@��I�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�$?   @�$?      �?!   @�$?) ��_;Y>2�[^:��"?U�4@@�$?�������:              �?        4X�t�       S�	/[b���A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) �P�Fr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)����7�+>2>h�'�?x?�x�?�������:              �?        ~�I��       S�	!7���A�*�
u
generator_loss_1*a	   ��I�?   ��I�?      �?!   ��I�?)@
lS�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �[�>    �[�>      �?!    �[�>) H�cu!	>2O�ʗ��>>�?�s��>�������:              �?        ��pX�       S�	8ͳ���A�*�
u
generator_loss_1*a	   �y��?   �y��?      �?!   �y��?) q&��Uq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `>�?   `>�?      �?!   `>�?)@*���>2��[�?1��a˲?�������:              �?        ��Ew�       S�	W����A�*�
u
generator_loss_1*a	   �C�?   �C�?      �?!   �C�?)@@���r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �w�>   �w�>      �?!   �w�>) D
�F��=2})�l a�>pz�w�7�>�������:              �?        �B��       S�	�7���A�*�
u
generator_loss_1*a	   �Ƌ�?   �Ƌ�?      �?!   �Ƌ�?) RG!3o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ` �?   ` �?      �?!   ` �?)@s�b6>2��d�r?�5�i}1?�������:              �?        �J"�       S�	������A�*�
u
generator_loss_1*a	   ��ȯ?   ��ȯ?      �?!   ��ȯ?) �-�)�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @2?   @2?      �?!   @2?) a���<>2�T7��?�vV�R9?�������:              �?        _�=��       S�	�d����A�*�
u
generator_loss_1*a	   @b�?   @b�?      �?!   @b�?) QC\�=r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ����>2I��P=�>��Zr[v�>�������:              �?        �����       S�	�����A�*�
u
generator_loss_1*a	   �H�?   �H�?      �?!   �H�?)@���2r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    sA�>    sA�>      �?!    sA�>) H�|=�>2>�?�s��>�FF�G ?�������:              �?        ��z��       S�	c�\ ��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) m�Dr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) �Z�1>2x?�x�?��d�r?�������:              �?        '�b��       �{�	Tp	��A*�
u
generator_loss_1*a	   ��/�?   ��/�?      �?!   ��/�?) `�_p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �3�?   �3�?      �?!   �3�?) I����D>2��ڋ?�.�?�������:              �?        #���       ۞��	t���A(*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@��B�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `�	?   `�	?      �?!   `�	?)@6P�R�>21��a˲?6�]��?�������:              �?        $�8��       ۞��	�ͻ��AP*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ���p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �*�>    �*�>      �?!    �*�>) Q>�j�=2�ߊ4F��>})�l a�>�������:              �?        ��}r�       ۞��	"��#��Ax*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)�,��}?n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) ����B>2�vV�R9?��ڋ?�������:              �?        �<U6�       S�	�Q],��A�*�
u
generator_loss_1*a	   ��`�?   ��`�?      �?!   ��`�?) �A��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @ֽ�>   @ֽ�>      �?!   @ֽ�>)�xn�'!>2��Zr[v�>O�ʗ��>�������:              �?        :�}�       S�	��>5��A�*�
u
generator_loss_1*a	    �	�?    �	�?      �?!    �	�?)  �(>$r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @��	?   @��	?      �?!   @��	?)�P�_��$>2����?f�ʜ�7
?�������:              �?        �B�#�       S�	P�>��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)  ٠�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�&?   @�&?      �?!   @�&?) �}���6>2��d�r?�5�i}1?�������:              �?        "y���       S�	 &�F��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@|S�Kr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��^�>   ��^�>      �?!   ��^�>) r��>2��Zr[v�>O�ʗ��>�������:              �?        R����       S�	ך�O��A�*�
u
generator_loss_1*a	   ��k�?   ��k�?      �?!   ��k�?)@x���p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @/��>   @/��>      �?!   @/��>)��e|OQ�=2�h���`�>�ߊ4F��>�������:              �?        sK��       S�	u��X��A�*�
u
generator_loss_1*a	    sİ?    sİ?      �?!    sİ?) ��R�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)�tv���I>2�.�?ji6�9�?�������:              �?        B���       S�	���a��A�*�
u
generator_loss_1*a	   ��Y�?   ��Y�?      �?!   ��Y�?) �INz�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �7�>   �7�>      �?!   �7�>)� �e$�=2��(���>a�Ϭ(�>�������:              �?        �	aP�       S�	�d�j��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) ��0�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�xIc`�=28K�ߝ�>�h���`�>�������:              �?        B���       S�	�6�s��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@�(��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �,��>   �,��>      �?!   �,��>) ī�j�=2})�l a�>pz�w�7�>�������:              �?        w3�'�       S�	&*�|��A�*�
u
generator_loss_1*a	   �S��?   �S��?      �?!   �S��?) ē�Kq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �C�>   �C�>      �?!   �C�>) Z�e`��=2a�Ϭ(�>8K�ߝ�>�������:              �?         ���       S�	6�����A�*�
u
generator_loss_1*a	   `�4�?   `�4�?      �?!   `�4�?)@ʎijp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �W�>   �W�>      �?!   �W�>)��i�y�=28K�ߝ�>�h���`�>�������:              �?        N���       S�	�ǎ��A�*�
u
generator_loss_1*a	    4z�?    4z�?      �?!    4z�?)  �Y�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) �<�8��=2��(���>a�Ϭ(�>�������:              �?        ��o�       S�	�_ї��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) @^qs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�'K�p�=2jqs&\��>��~]�[�>�������:              �?        �o��       S�	MӠ��A�*�
u
generator_loss_1*a	   @�î?   @�î?      �?!   @�î?)�hy]�m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @ �>   @ �>      �?!   @ �>) �Tt��=2�uE����>�f����>�������:              �?        W)�L�       S�	1`���A�*�
u
generator_loss_1*a	   @r°?   @r°?      �?!   @r°?) ѿ��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �%��>   �%��>      �?!   �%��>) ��D��=2�f����>��(���>�������:              �?        Ľ�'�       S�	1���A�*�
u
generator_loss_1*a	   �)~�?   �)~�?      �?!   �)~�?)@J6 q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���$?   ���$?      �?!   ���$?) �4T ^[>2U�4@@�$?+A�F�&?�������:              �?        ]���       S�	������A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) �C��:r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @x%�>   @x%�>      �?!   @x%�>)����f�=2�iD*L��>E��a�W�>�������:              �?        "_�       S�	!���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) �IaK-p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @%
 ?   @%
 ?      �?!   @%
 ?) ���P>2>�?�s��>�FF�G ?�������:              �?        �	]<�       �{�	7e	���A*�
u
generator_loss_1*a	    $ӱ?    $ӱ?      �?!    $ӱ?)  �Ŏ�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	     ��>     ��>      �?!     ��>)   @Fx6=2�4[_>��>
�}���>�������:              �?        �"_��       ۞��	�'���A(*�
u
generator_loss_1*a	   `�9�?   `�9�?      �?!   `�9�?)@T�q�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @'��>   @'��>      �?!   @'��>)�$.�C=2.��fc��>39W$:��>�������:              �?        ^n�       ۞��	�0i���AP*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@~�Mr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    a��>    a��>      �?!    a��>) ���W=2��|�~�>���]���>�������:              �?        )���       ۞��	�����Ax*�
u
generator_loss_1*a	    E��?    E��?      �?!    E��?)@��n�)q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ���=2�*��ڽ>�[�=�k�>�������:              �?        A�]�       S�	�����A�*�
u
generator_loss_1*a	   �&=�?   �&=�?      �?!   �&=�?) ��6{p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��Q�>   ��Q�>      �?!   ��Q�>) �jP�>2>�?�s��>�FF�G ?�������:              �?        �* ��       S�	�=����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@��Ь5s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    }[�>    }[�>      �?!    }[�>) H��G�>2>�?�s��>�FF�G ?�������:              �?        ƂV�       S�	8�/��A�*�
u
generator_loss_1*a	   ��A�?   ��A�?      �?!   ��A�?)@>Ю܄p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@�{Cԍ>2�FF�G ?��[�?�������:              �?        ��QL�       S�	g��A�*�
u
generator_loss_1*a	   �ك�?   �ك�?      �?!   �ك�?) q;	�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �Z�=2E��a�W�>�ѩ�-�>�������:              �?        �R��       S�	7E���A�*�
u
generator_loss_1*a	   ��V�?   ��V�?      �?!   ��V�?)��_���l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��m�>   ��m�>      �?!   ��m�>) �T�Ĳ�=2�f����>��(���>�������:              �?        ����       S�	��� ��A�*�
u
generator_loss_1*a	    �Ͱ?    �Ͱ?      �?!    �Ͱ?) ����q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  kE��=2���%�>�uE����>�������:              �?        �{��       S�	�2*��A�*�
u
generator_loss_1*a	    �8�?    �8�?      �?!    �8�?)@d�;3rp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) d�h؁�=2���%�>�uE����>�������:              �?        �J!��       S�	���3��A�*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?) ����o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    aZ�>    aZ�>      �?!    aZ�>)@��2�h�=2�uE����>�f����>�������:              �?        �ӓQ�       S�	��<��A�*�
u
generator_loss_1*a	   �6��?   �6��?      �?!   �6��?) Y�P[pq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �:��>   �:��>      �?!   �:��>) �O��=2�f����>��(���>�������:              �?        �F��       S�	�AF��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) B�G�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �n�>    �n�>      �?!    �n�>) �а��=2E��a�W�>�ѩ�-�>�������:              �?        H�KG�       S�	:�UO��A�*�
u
generator_loss_1*a	    ٰ?    ٰ?      �?!    ٰ?) Y��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    DV�>    DV�>      �?!    DV�>)  !/�=2�f����>��(���>�������:              �?        �����       S�	��X��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ��{}�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @n��>   @n��>      �?!   @n��>)�؏�,��=2E��a�W�>�ѩ�-�>�������:              �?        ��c2�       S�	�b��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@$�e�-p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �H�>   �H�>      �?!   �H�>) R3U���=2�iD*L��>E��a�W�>�������:              �?        b�9��       S�	�Nzk��A�*�
u
generator_loss_1*a	   @�;�?   @�;�?      �?!   @�;�?) !�Y��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �T�>   �T�>      �?!   �T�>) !�����=2���%�>�uE����>�������:              �?        @o*
�       S�	2��t��A�*�
u
generator_loss_1*a	    {�?    {�?      �?!    {�?) Ș��o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@�-�#>>2�T7��?�vV�R9?�������:              �?        �66��       S�	�#~��A�*�
u
generator_loss_1*a	   `,��?   `,��?      �?!   `,��?)@7�R/s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �]�>   �]�>      �?!   �]�>)@�ˌ�r�=2��>M|K�>�_�T�l�>�������:              �?        ��G�       S�	������A�*�
u
generator_loss_1*a	    �i�?    �i�?      �?!    �i�?)@� �:�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�|�>   @�|�>      �?!   @�|�>) %:��=2�ѩ�-�>���%�>�������:              �?        ��:�       S�	����A�*�
u
generator_loss_1*a	   �l��?   �l��?      �?!   �l��?) įwL0q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `�B�>   `�B�>      �?!   `�B�>)@�XX@�=2�f����>��(���>�������:              �?        6�z$�       �{�	��G���A*�
u
generator_loss_1*a	   �<R�?   �<R�?      �?!   �<R�?)@�M��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �[��>   �[��>      �?!   �[��>) ��[Lo�=2a�Ϭ(�>8K�ߝ�>�������:              �?        �g�?�       ۞��	ު����A(*�
u
generator_loss_1*a	   ��_�?   ��_�?      �?!   ��_�?) ѝ%��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    Z�>    Z�>      �?!    Z�>)  U2��=28K�ߝ�>�h���`�>�������:              �?        aal�       ۞��	wS���AP*�
u
generator_loss_1*a	   �|�?   �|�?      �?!   �|�?)@��67?p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �h��>   �h��>      �?!   �h��>) ���=2��(���>a�Ϭ(�>�������:              �?        �V���       ۞��	��۶��Ax*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) �_"�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �\�>   �\�>      �?!   �\�>)@���*��=2�ߊ4F��>})�l a�>�������:              �?        �����       S�	�[���A�*�
u
generator_loss_1*a	   ��2�?   ��2�?      �?!   ��2�?)@ȼv]fp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��.�>   ��.�>      �?!   ��.�>)@vC��=2�ߊ4F��>})�l a�>�������:              �?        ��Ť�       S�	o�����A�*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?)@Ȍ'%q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   � W?   � W?      �?!   � W?) �(J�,>2>h�'�?x?�x�?�������:              �?        ��.��       S�	2�V���A�*�
u
generator_loss_1*a	   �?   �?      �?!   �?) $�*aq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    [��>    [��>      �?!    [��>) ~��J>2��Zr[v�>O�ʗ��>�������:              �?        ��8y�       S�	h����A�*�
u
generator_loss_1*a	    �H�?    �H�?      �?!    �H�?)  @���r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �R�>    �R�>      �?!    �R�>) ��LU��=2�ѩ�-�>���%�>�������:              �?        C���       S�	k�W���A�*�
u
generator_loss_1*a	   @)��?   @)��?      �?!   @)��?) Y�=�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@ �U��=2�uE����>�f����>�������:              �?        ���#�       S�	�����A�*�
u
generator_loss_1*a	    !�?    !�?      �?!    !�?) D�jCr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �^��>   �^��>      �?!   �^��>)@�v���=2���%�>�uE����>�������:              �?        Z��       S�	�i����A�*�
u
generator_loss_1*a	   ��\�?   ��\�?      �?!   ��\�?) �OI��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�8�>   @�8�>      �?!   @�8�>) ��Kr�=2�ѩ�-�>���%�>�������:              �?        �%d��       S�	p!��A�*�
u
generator_loss_1*a	    {(�?    {(�?      �?!    {(�?) ��áVn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    Y��>    Y��>      �?!    Y��>) /E?V�=2�f����>��(���>�������:              �?        �V���       S�	����A�*�
u
generator_loss_1*a	   �m�?   �m�?      �?!   �m�?)@����!r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�P�>   @�P�>      �?!   @�P�>) ���LQ�=2�uE����>�f����>�������:              �?        3.�
�       S�	4}��A�*�
u
generator_loss_1*a	   �o6�?   �o6�?      �?!   �o6�?) �\�f�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �X�>   �X�>      �?!   �X�>)@�!<f�=2��~]�[�>��>M|K�>�������:              �?        �T>,�       S�	�� ��A�*�
u
generator_loss_1*a	   ��a�?   ��a�?      �?!   ��a�?) ɨd�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @u.?   @u.?      �?!   @u.?) 9��t>21��a˲?6�]��?�������:              �?        8���       S�	�@�)��A�*�
u
generator_loss_1*a	   `;R�?   `;R�?      �?!   `;R�?)@Vta�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) yQ[���=2�h���`�>�ߊ4F��>�������:              �?        A�Đ�       S�	�Le3��A�*�
u
generator_loss_1*a	   �N�?   �N�?      �?!   �N�?) $�9~�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `	4�>   `	4�>      �?!   `	4�>) �zp$ �=2a�Ϭ(�>8K�ߝ�>�������:              �?        �����       S�	�=��A�*�
u
generator_loss_1*a	    ï?    ï?      �?!    ï?) $�x�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �$�>    �$�>      �?!    �$�>) pk���=2��(���>a�Ϭ(�>�������:              �?        ��       S�	��F��A�*�
u
generator_loss_1*a	   �L�?   �L�?      �?!   �L�?) č}�6r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @G�>   @G�>      �?!   @G�>) ��Lò�=2��~]�[�>��>M|K�>�������:              �?        ��c?�       S�	��|P��A�*�
u
generator_loss_1*a	   ��{�?   ��{�?      �?!   ��{�?)@Dd�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `D~�>   `D~�>      �?!   `D~�>) �X��=2�_�T�l�>�iD*L��>�������:              �?        CZ��       S�	 �6Z��A�*�
u
generator_loss_1*a	   ��x�?   ��x�?      �?!   ��x�?) ����n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��M�>   ��M�>      �?!   ��M�>) p�ғ��=2E��a�W�>�ѩ�-�>�������:              �?        ����       S�	���c��A�*�
u
generator_loss_1*a	   `�b�?   `�b�?      �?!   `�b�?)@�4��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �U�>    �U�>      �?!    �U�>) g��=2���%�>�uE����>�������:              �?        �0���       �{�	a��m��A*�
u
generator_loss_1*a	   `s\�?   `s\�?      �?!   `s\�?) ��*�n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) H
�=��=2�iD*L��>E��a�W�>�������:              �?        D��       ۞��	��_w��A(*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@P7A�4q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��9�>   ��9�>      �?!   ��9�>)��&��=2
�/eq
�>;�"�q�>�������:              �?        ����       ۞��	]�4���AP*�
u
generator_loss_1*a	   ��6�?   ��6�?      �?!   ��6�?)@x�n�np?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    O��>    O��>      �?!    O��>) �<���=2jqs&\��>��~]�[�>�������:              �?        ��G�       ۞��	������Ax*�
u
generator_loss_1*a	   �uH�?   �uH�?      �?!   �uH�?) �"�3�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    d�>    d�>      �?!    d�>) H���ަ=2;�"�q�>['�?��>�������:              �?        $+ΰ�       S�	�!Ĕ��A�*�
u
generator_loss_1*a	   @͋�?   @͋�?      �?!   @͋�?) ��`q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    7��>    7��>      �?!    7��>)@��Y��=2K+�E���>jqs&\��>�������:              �?        !0a�       S�	"�����A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?)@I��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �	��>   �	��>      �?!   �	��>) �ŜT�=2K+�E���>jqs&\��>�������:              �?        ��sU�       S�	�t���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@(2s��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �f�nĭ=2['�?��>K+�E���>�������:              �?        �{	�       S�	��K���A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) G�Hq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �!��>   �!��>      �?!   �!��>) "r�=2�_�T�l�>�iD*L��>�������:              �?         ����       S�	}�.���A�*�
u
generator_loss_1*a	   �Ɐ?   �Ɐ?      �?!   �Ɐ?) �n�do?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @0#�>   @0#�>      �?!   @0#�>) ����=2��~]�[�>��>M|K�>�������:              �?        ÷ <�       S�	��'���A�*�
u
generator_loss_1*a	   �ԯ?   �ԯ?      �?!   �ԯ?)�0��c�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @^�>   @^�>      �?!   @^�>)����v�=2�*��ڽ>�[�=�k�>�������:              �?        ���4�       S�	�L���A�*�
u
generator_loss_1*a	   �Z��?   �Z��?      �?!   �Z��?) ���o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) B����=2�iD*L��>E��a�W�>�������:              �?        ����       S�	��M���A�*�
u
generator_loss_1*a	   `]M�?   `]M�?      �?!   `]M�?)@���0�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �_>�>   �_>�>      �?!   �_>�>)@�S7✹=2��~]�[�>��>M|K�>�������:              �?        Soz�       S�	,Q���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@�9"��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) Y8
��=2��~���>�XQ��>�������:              �?        @)b��       S�	Xk=���A�*�
u
generator_loss_1*a	    �o�?    �o�?      �?!    �o�?) D2��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �4��>   �4��>      �?!   �4��>) D\7��=2�XQ��>�����>�������:              �?        :�<��       S�	֖'���A�*�
u
generator_loss_1*a	    N��?    N��?      �?!    N��?)@x��rq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�P�>   @�P�>      �?!   @�P�>) �=�s��=2�*��ڽ>�[�=�k�>�������:              �?        Ò���       S�	��2��A�*�
u
generator_loss_1*a	   �I��?   �I��?      �?!   �I��?)@	�
q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��f�>   ��f�>      �?!   ��f�>) 2�g�ȥ=2
�/eq
�>;�"�q�>�������:              �?        �h��       S�	��-��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@dڳCr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    Ca�>    Ca�>      �?!    Ca�>) �~Iq��=2G&�$�>�*��ڽ>�������:              �?        _���       S�	��/��A�*�
u
generator_loss_1*a	   `"��?   `"��?      �?!   `"��?)@�q��Bs?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �L��>   �L��>      �?!   �L��>)�X��{�=25�"�g��>G&�$�>�������:              �?        �vpG�       S�	.�$ ��A�*�
u
generator_loss_1*a	   �.H�?   �.H�?      �?!   �.H�?) $���p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �KѸ>   �KѸ>      �?!   �KѸ>) "�"J?�=25�"�g��>G&�$�>�������:              �?        �a�       S�	!
!*��A�*�
u
generator_loss_1*a	    J��?    J��?      �?!    J��?)@h���_s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��û>   ��û>      �?!   ��û>) ���?�=2G&�$�>�*��ڽ>�������:              �?        %1)��       S�	W8 4��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) �hEn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ]��>    ]��>      �?!    ]��>) }B�M�=2E��a�W�>�ѩ�-�>�������:              �?        q�S�       S�	�>��A�*�
u
generator_loss_1*a	   ��F�?   ��F�?      �?!   ��F�?)@$�@�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �FT�>   �FT�>      �?!   �FT�>) �@�Ē=2�[�=�k�>��~���>�������:              �?        t����       �{�	��H��A*�
u
generator_loss_1*a	   ��Ư?   ��Ư?      �?!   ��Ư?)�,N���o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �r^�;�=25�"�g��>G&�$�>�������:              �?        �w�\�       ۞��	�.R��A(*�
u
generator_loss_1*a	   @�'�?   @�'�?      �?!   @�'�?) i�0er?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `�d�>   `�d�>      �?!   `�d�>)@����=2K+�E���>jqs&\��>�������:              �?        ��q`�       ۞��	��H\��AP*�
u
generator_loss_1*a	   ��°?   ��°?      �?!   ��°?)@�|�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)�v�_�=2['�?��>K+�E���>�������:              �?        �����       ۞��	��[f��Ax*�
u
generator_loss_1*a	    C��?    C��?      �?!    C��?) Hl�%m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) h�I �=2�_�T�l�>�iD*L��>�������:              �?        ;J�Z�       S�	x-�p��A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) 9��7r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  �3��=25�"�g��>G&�$�>�������:              �?        p��6�       S�	�=�z��A�*�
u
generator_loss_1*a	   �ȯ?   �ȯ?      �?!   �ȯ?) �6\��o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `HM�>   `HM�>      �?!   `HM�>)@b�	��=2�*��ڽ>�[�=�k�>�������:              �?        �9��       S�	.�ӄ��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) ��,!l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    #��>    #��>      �?!    #��>) ���`B�=2�[�=�k�>��~���>�������:              �?        �]�	�       S�	�#���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) 7r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �n��>   �n��>      �?!   �n��>)�L�=sa=2�5�L�>;9��R�>�������:              �?        EԸ<�       S�	5+���A�*�
u
generator_loss_1*a	    �ϯ?    �ϯ?      �?!    �ϯ?)  �c]�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) 2�D�v�=28K�ߝ�>�h���`�>�������:              �?        �!v��       S�	E8c���A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) `�=k?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   �#��>   �#��>      �?!   �#��>)��O���=2�iD*L��>E��a�W�>�������:              �?        ���       S�	�ȏ���A�*�
u
generator_loss_1*a	    �ǰ?    �ǰ?      �?!    �ǰ?) @�VJ�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @o��>   @o��>      �?!   @o��>) ���³�=2��~]�[�>��>M|K�>�������:              �?        ?-7��       S�	A�ʷ��A�*�
u
generator_loss_1*a	   @9a�?   @9a�?      �?!   @9a�?) ��F��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �B�=2��~]�[�>��>M|K�>�������:              �?        ��q#�       S�	�����A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)  [6��o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)���W��=2�_�T�l�>�iD*L��>�������:              �?        6:�?�       S�	$cS���A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)  D��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �{��>   �{��>      �?!   �{��>) �4�+�=2['�?��>K+�E���>�������:              �?        �5�U�       S�	˧����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) 1��!�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@�5^� �=2��~���>�XQ��>�������:              �?        ۥ��       S�	������A�*�
u
generator_loss_1*a	   ��j�?   ��j�?      �?!   ��j�?)�,~��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    g��>    g��>      �?!    g��>) V����=2�iD*L��>E��a�W�>�������:              �?        U{��       S�	6`G���A�*�
u
generator_loss_1*a	   ��w�?   ��w�?      �?!   ��w�?) �Tn^�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) V)�.M�=2;�"�q�>['�?��>�������:              �?        RF��       S�	�ћ���A�*�
u
generator_loss_1*a	   ��D�?   ��D�?      �?!   ��D�?) =A�I�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �b	�>   �b	�>      �?!   �b	�>) 2�HY�=2;�"�q�>['�?��>�������:              �?        {	it�       S�	GB����A�*�
u
generator_loss_1*a	    M��?    M��?      �?!    M��?) �2��Ts?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�/�>   @�/�>      �?!   @�/�>)���G�G�=2
�/eq
�>;�"�q�>�������:              �?        ����       S�	w�N
��A�*�
u
generator_loss_1*a	   �$�?   �$�?      �?!   �$�?) D�mm0p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @'��>   @'��>      �?!   @'��>) I�d �=2��~���>�XQ��>�������:              �?        0���       S�	p����A�*�
u
generator_loss_1*a	   �A��?   �A��?      �?!   �A��?) ���o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `�&�>   `�&�>      �?!   `�&�>)@&�ԋM�=2K+�E���>jqs&\��>�������:              �?        �d6��       S�	6_��A�*�
u
generator_loss_1*a	   `���?   `���?      �?!   `���?)@��&0t?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	    }��>    }��>      �?!    }��>) ����=2�����>
�/eq
�>�������:              �?        ��%�       �{�	�.X)��A*�
u
generator_loss_1*a	   @9�?   @9�?      �?!   @9�?) ��5�rp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �om��=2jqs&\��>��~]�[�>�������:              �?        !��+�       ۞��	�o�3��A(*�
u
generator_loss_1*a	   `y��?   `y��?      �?!   `y��?)@����Yq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��K�>   ��K�>      �?!   ��K�>)@V̄��=2�����>
�/eq
�>�������:              �?        ���-�       ۞��	~�!>��AP*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?)@�E":p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `�&�>   `�&�>      �?!   `�&�>)@&[��b�=2�[�=�k�>��~���>�������:              �?        �ta��       ۞��	�K�H��Ax*�
u
generator_loss_1*a	   �ծ?   �ծ?      �?!   �ծ?) 8ݔ%�m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��d�>   ��d�>      �?!   ��d�>)��)~��j=2���?�ګ>����>�������:              �?        Jyz�       S�	�S��A�*�
u
generator_loss_1*a	   `vq�?   `vq�?      �?!   `vq�?)@�w'@s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @i˲>   @i˲>      �?!   @i˲>) Yl��v=2��n����>�u`P+d�>�������:              �?        "��       S�	3��]��A�*�
u
generator_loss_1*a	   ��ڱ?   ��ڱ?      �?!   ��ڱ?)@6�7�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �4�>    �4�>      �?!    �4�>) \���Ӏ=20�6�/n�>5�"�g��>�������:              �?        �h�|�       S�	nVh��A�*�
u
generator_loss_1*a	   @f�?   @f�?      �?!   @f�?) q�8�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ,��>    ,��>      �?!    ,��>) �����=2�����>
�/eq
�>�������:              �?        ��j�       S�	��r��A�*�
u
generator_loss_1*a	    牱?    牱?      �?!    牱?) �p�9s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��6�>   ��6�>      �?!   ��6�>) $��^n�=2�*��ڽ>�[�=�k�>�������:              �?        c*���       S�	��S}��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) a�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@�צ���=2��~]�[�>��>M|K�>�������:              �?        �!C��       S�	�4܇��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@�'dyq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �k��>   �k��>      �?!   �k��>)@�ǰ���=2��~���>�XQ��>�������:              �?        +bo��       S�	��V���A�*�
u
generator_loss_1*a	   `�V�?   `�V�?      �?!   `�V�?)@.VK�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �킽>   �킽>      �?!   �킽>) ��cf7�=2G&�$�>�*��ڽ>�������:              �?        �@�       S�	�8���A�*�
u
generator_loss_1*a	   @�6�?   @�6�?      �?!   @�6�?)����l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ���(ٖ=2��~���>�XQ��>�������:              �?        �t��       S�	�#}���A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) �^p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �L!�>   �L!�>      �?!   �L!�>) ).�Sy=2��n����>�u`P+d�>�������:              �?        ����       S�	E/���A�*�
u
generator_loss_1*a	   �e��?   �e��?      �?!   �e��?) �.3x-m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�D�Ը��=2;�"�q�>['�?��>�������:              �?        !G�_�       S�	=�����A�*�
u
generator_loss_1*a	   @G�?   @G�?      �?!   @G�?) I5[�'r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�����g�=2�����>
�/eq
�>�������:              �?        ��·�       S�	��>���A�*�
u
generator_loss_1*a	   `�i�?   `�i�?      �?!   `�i�?)@�!��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �oS�>   �oS�>      �?!   �oS�>) ���7�=2
�/eq
�>;�"�q�>�������:              �?        ��y�       S�	(a����A�*�
u
generator_loss_1*a	   `Q��?   `Q��?      �?!   `Q��?) ﲯ�<m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ��RX��=2��~���>�XQ��>�������:              �?        ��*�       S�	zʝ���A�*�
u
generator_loss_1*a	   �	��?   �	��?      �?!   �	��?) �M>#q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) v(!ő=2�[�=�k�>��~���>�������:              �?        ��p�       S�	.�J���A�*�
u
generator_loss_1*a	   �SW�?   �SW�?      �?!   �SW�?)@��a��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �~A�>   �~A�>      �?!   �~A�>)��@��=20�6�/n�>5�"�g��>�������:              �?        �b�c�       S�	�����A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@\5ŢDr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �>   �>      �?!   �>) ����|=2�u`P+d�>0�6�/n�>�������:              �?        :��}�       S�	7>����A�*�
u
generator_loss_1*a	   �!<�?   �!<�?      �?!   �!<�?) 1g}%yp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �zެJ{=2�u`P+d�>0�6�/n�>�������:              �?        ���z�       S�	)�o��A�*�
u
generator_loss_1*a	   ��ٰ?   ��ٰ?      �?!   ��ٰ?)@�=�ƾq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `%��>   `%��>      �?!   `%��>) �7��݂=20�6�/n�>5�"�g��>�������:              �?        ߸af�       �{�	�	��A*�
u
generator_loss_1*a	   @.��?   @.��?      �?!   @.��?) ��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��:�>   ��:�>      �?!   ��:�>)@>�6!�~=2�u`P+d�>0�6�/n�>�������:              �?        �a8�       ۞��	p����A(*�
u
generator_loss_1*a	   �v��?   �v��?      �?!   �v��?)@4��sMq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �R�>   �R�>      �?!   �R�>) lɘUw=2��n����>�u`P+d�>�������:              �?        ˙�$�       ۞��	�*v'��AP*�
u
generator_loss_1*a	    
h�?    
h�?      �?!    
h�?) @¹�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��]�>   ��]�>      �?!   ��]�>) ShU%i=2���?�ګ>����>�������:              �?        ��M�       ۞��	P�*2��Ax*�
u
generator_loss_1*a	    A��?    A��?      �?!    A��?) ԛ��o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @ɔ�>   @ɔ�>      �?!   @ɔ�>)����`�=2�����>
�/eq
�>�������:              �?        �Ʌ��       S�	,��<��A�*�
u
generator_loss_1*a	   `�x�?   `�x�?      �?!   `�x�?)@J%a�s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    W�>    W�>      �?!    W�>)@lN���v=2��n����>�u`P+d�>�������:              �?        L����       S�	�&�G��A�*�
u
generator_loss_1*a	   @�]�?   @�]�?      �?!   @�]�?) iWS�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) ���~>{=2�u`P+d�>0�6�/n�>�������:              �?        <�t;�       S�	eA�R��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ��� r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��F�>   ��F�>      �?!   ��F�>) 5y�e�l=2���?�ګ>����>�������:              �?        ݗ�       S�	f�G]��A�*�
u
generator_loss_1*a	    �Ͱ?    �Ͱ?      �?!    �Ͱ?)@��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �f	�>   �f	�>      �?!   �f	�>) [v�h=2���?�ګ>����>�������:              �?        �XQ�       S�	��)h��A�*�
u
generator_loss_1*a	   ��a�?   ��a�?      �?!   ��a�?) A�R��r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�z�>   @�z�>      �?!   @�z�>)��}q4�b=2�5�L�>;9��R�>�������:              �?        �����       S�	��s��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) ,��&m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `�3�>   `�3�>      �?!   `�3�>)@����~r=2豪}0ڰ>��n����>�������:              �?        .Eo��       S�	�J�}��A�*�
u
generator_loss_1*a	    #�?    #�?      �?!    #�?) Hv!��m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    Kѻ>    Kѻ>      �?!    Kѻ>) �_l�.�=2G&�$�>�*��ڽ>�������:              �?        \'~��       S�	�~����A�*�
u
generator_loss_1*a	   `�I�?   `�I�?      �?!   `�I�?)@v��5�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    }��>    }��>      �?!    }��>) ��wa�=2��>M|K�>�_�T�l�>�������:              �?        C*�t�       S�	����A�*�
u
generator_loss_1*a	    �k�?    �k�?      �?!    �k�?) ����p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @;�>   @;�>      �?!   @;�>)��Cc�=20�6�/n�>5�"�g��>�������:              �?        lt{��       S�	]䯞��A�*�
u
generator_loss_1*a	    m�?    m�?      �?!    m�?) �����q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �>    �>      �?!    �>) �U>�d=2;9��R�>���?�ګ>�������:              �?        Ș<&�       S�	�䍩��A�*�
u
generator_loss_1*a	    �F�?    �F�?      �?!    �F�?)@�d�k�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ���Qv=2��n����>�u`P+d�>�������:              �?        ?�D�       S�	��m���A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@��!�lq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �냙>   �냙>      �?!   �냙>) "��?XD=2.��fc��>39W$:��>�������:              �?        ��aI�       S�	�zd���A�*�
u
generator_loss_1*a	   �z*�?   �z*�?      �?!   �z*�?)@���jr?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) d��Y^=2���]���>�5�L�>�������:              �?        w˵��       S�	��]���A�*�
u
generator_loss_1*a	   @�۰?   @�۰?      �?!   @�۰?) ��{�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �$t�>   �$t�>      �?!   �$t�>)@�c7+�|=2�u`P+d�>0�6�/n�>�������:              �?        �.��       S�	W�R���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) D��V-p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �e��>   �e��>      �?!   �e��>) �`���=2G&�$�>�*��ڽ>�������:              �?        y:jj�       S�	� ]���A�*�
u
generator_loss_1*a	   �|�?   �|�?      �?!   �|�?) N���3n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    B=�>    B=�>      �?!    B=�>)  (��/�=2;�"�q�>['�?��>�������:              �?        �}sP�       S�	bS����A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)@$�T�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) {�\�=2['�?��>K+�E���>�������:              �?        2ci��       S�	�����A�*�
u
generator_loss_1*a	    @߭?    @߭?      �?!    @߭?)  ����k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    [�>    [�>      �?!    [�>)@l-K#��=2��>M|K�>�_�T�l�>�������:              �?        v]{�       �{�	�{���A*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)  �	�=p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)��{&�a�=2;�"�q�>['�?��>�������:              �?        ���k�       ۞��	U'���A(*�
u
generator_loss_1*a	   @ӹ�?   @ӹ�?      �?!   @ӹ�?)��0��m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) D��W>2��Zr[v�>O�ʗ��>�������:              �?        �6E �       ۞��	�����AP*�
u
generator_loss_1*a	    D��?    D��?      �?!    D��?) ��g��k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@"�ˈ�>21��a˲?6�]��?�������:              �?        -�O{�       ۞��	ap�"��Ax*�
u
generator_loss_1*a	   �?��?   �?��?      �?!   �?��?) �+��bm?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) ��Zâ�=2})�l a�>pz�w�7�>�������:              �?        QP��       S�	+��-��A�*�
u
generator_loss_1*a	   @u�?   @u�?      �?!   @u�?)�h~�.�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  $���=2�h���`�>�ߊ4F��>�������:              �?        �ay��       S�	�)�8��A�*�
u
generator_loss_1*a	   �ᣰ?   �ᣰ?      �?!   �ᣰ?) $��QNq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    RO?    RO?      �?!    RO?)@����>2�FF�G ?��[�?�������:              �?        �*i	�       S�	aXD��A�*�
u
generator_loss_1*a	   ��6�?   ��6�?      �?!   ��6�?) ��rn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �-��>   �-��>      �?!   �-��>) �"�%��=2pz�w�7�>I��P=�>�������:              �?        �OG�       S�	s3O��A�*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?)@B����q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `�d�>   `�d�>      �?!   `�d�>)@~ ��=2�h���`�>�ߊ4F��>�������:              �?        ��[�       S�	�^Z��A�*�
u
generator_loss_1*a	   �Yj�?   �Yj�?      �?!   �Yj�?) q'���r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)��%�ه�=2a�Ϭ(�>8K�ߝ�>�������:              �?        Hݘ�       S�	�̙e��A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?)�hz�$�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) R��'k�=2��(���>a�Ϭ(�>�������:              �?         ����       S�	p��p��A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?)����L�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �O?    �O?      �?!    �O?) Ta���>2k�1^�sO?nK���LQ?�������:              �?        ��
��       S�	kG|��A�*�
u
generator_loss_1*a	   �ha�?   �ha�?      �?!   �ha�?) �9��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `1�?   `1�?      �?!   `1�?) /���\!>26�]��?����?�������:              �?        ��b�       S�	`�P���A�*�
u
generator_loss_1*a	    �:�?    �:�?      �?!    �:�?) ��} vp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ��?    ��?      �?!    ��?)@�Pb�1>2x?�x�?��d�r?�������:              �?        WP���       S�	�H����A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)@���Np?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��0�>   ��0�>      �?!   ��0�>) �n^WI>2I��P=�>��Zr[v�>�������:              �?        ����       S�	Z����A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) D��Lq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �?��>   �?��>      �?!   �?��>)@����=2�f����>��(���>�������:              �?        $��Z�       S�	��A���A�*�
u
generator_loss_1*a	   �x�?   �x�?      �?!   �x�?) I7��p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �c�>   �c�>      �?!   �c�>) 1����=2�uE����>�f����>�������:              �?        �L�<�       S�	����A�*�
u
generator_loss_1*a	   �*��?   �*��?      �?!   �*��?) �l��qm?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  ��3��=2��(���>a�Ϭ(�>�������:              �?        6�߆�       S�	^����A�*�
u
generator_loss_1*a	   �0��?   �0��?      �?!   �0��?) �3X#co?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    6�>    6�>      �?!    6�>)@硐�=2})�l a�>pz�w�7�>�������:              �?        �R�       S�	L�/���A�*�
u
generator_loss_1*a	   �n�?   �n�?      �?!   �n�?)����n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��F�>   ��F�>      �?!   ��F�>) D��=2�h���`�>�ߊ4F��>�������:              �?        ���       S�	�|���A�*�
u
generator_loss_1*a	    ڣ�?    ڣ�?      �?!    ڣ�?) @Z�ANq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �UHe�=2�f����>��(���>�������:              �?        �����       S�	u����A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) w��[Do?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �?    �?      �?!    �?)@y�~�1>2x?�x�?��d�r?�������:              �?        ���       S�	an&���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)�d�D{�m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    l�?    l�?      �?!    l�?)  ���t>2��[�?1��a˲?�������:              �?        ����       �{�	�Hn���A*�
u
generator_loss_1*a	   �8�?   �8�?      �?!   �8�?)�!��&g?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)��j�(>2f�ʜ�7
?>h�'�?�������:              �?        e�r��       ۞��	U+���A(*�
u
generator_loss_1*a	    �0�?    �0�?      �?!    �0�?) ����{l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @)��>   @)��>      �?!   @)��>)�,����>2��Zr[v�>O�ʗ��>�������:              �?        ��%��       ۞��	~<��AP*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) 9��Kq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @Rg�>   @Rg�>      �?!   @Rg�>)�hO�H�>2I��P=�>��Zr[v�>�������:              �?        u����       ۞��	y����Ax*�
u
generator_loss_1*a	   �z$�?   �z$�?      �?!   �z$�?) �t��Nn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) �9��>2>�?�s��>�FF�G ?�������:              �?        ��U��       S�	�A�%��A�*�
u
generator_loss_1*a	   ��O�?   ��O�?      �?!   ��O�?)�`Ҋ��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@8��k�=2})�l a�>pz�w�7�>�������:              �?        �g� �       S�	��1��A�*�
u
generator_loss_1*a	   �s�?   �s�?      �?!   �s�?) �aZ�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��Y�>   ��Y�>      �?!   ��Y�>) �o���=2a�Ϭ(�>8K�ߝ�>�������:              �?        �M��       S�	�=��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)  ��rLm?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) �DM��=2�uE����>�f����>�������:              �?        3���       S�	�)�H��A�*�
u
generator_loss_1*a	    %�?    %�?      �?!    %�?) Ț��<n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �Ĕ�>   �Ĕ�>      �?!   �Ĕ�>) >Ӎs�=2a�Ϭ(�>8K�ߝ�>�������:              �?        �su�       S�	Hq�S��A�*�
u
generator_loss_1*a	   @wխ?   @wխ?      �?!   @wխ?)�d@!x�k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @�}�>   @�}�>      �?!   @�}�>) Yk��^�=2})�l a�>pz�w�7�>�������:              �?        �,X9�       S�	�F^_��A�*�
u
generator_loss_1*a	   `9��?   `9��?      �?!   `9��?)@��?�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)� V�o�=28K�ߝ�>�h���`�>�������:              �?        �ִ��       S�	���j��A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?)���,#m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) gI&��=2��(���>a�Ϭ(�>�������:              �?        �ս�       S�	NI`v��A�*�
u
generator_loss_1*a	    �z�?    �z�?      �?!    �z�?)  ��w�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @ o�>   @ o�>      �?!   @ o�>) x��<�=2���%�>�uE����>�������:              �?        ?��~�       S�	S����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) 9�u��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �� �>   �� �>      �?!   �� �>) ���U�=2�ѩ�-�>���%�>�������:              �?        Oa���       S�	������A�*�
u
generator_loss_1*a	    d��?    d��?      �?!    d��?) ��+��m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ��>2O�ʗ��>>�?�s��>�������:              �?        ��t��       S�	1����A�*�
u
generator_loss_1*a	   ��n�?   ��n�?      �?!   ��n�?)@!Q��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) 
����=2a�Ϭ(�>8K�ߝ�>�������:              �?        t%���       S�	�Ⲥ��A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?) �c�4p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `T&�>   `T&�>      �?!   `T&�>)@��.`�=2�uE����>�f����>�������:              �?        ҫ#]�       S�	H�O���A�*�
u
generator_loss_1*a	   �U�?   �U�?      �?!   �U�?) ���0�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  �rO\�=2�ѩ�-�>���%�>�������:              �?        ~Aҧ�       S�	����A�*�
u
generator_loss_1*a	   �:��?   �:��?      �?!   �:��?)@�D�~@q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)���t%�=2��(���>a�Ϭ(�>�������:              �?        �_T��       S�	������A�*�
u
generator_loss_1*a	   @pa�?   @pa�?      �?!   @pa�?)��.��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)@X�4��=2�h���`�>�ߊ4F��>�������:              �?        �����       S�	�K"���A�*�
u
generator_loss_1*a	   `�A�?   `�A�?      �?!   `�A�?)@�z1�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �V��>   �V��>      �?!   �V��>) ��&�=2�_�T�l�>�iD*L��>�������:              �?        �/���       S�	�s����A�*�
u
generator_loss_1*a	   `�ڮ?   `�ڮ?      �?!   `�ڮ?) �V]I�m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @c�>   @c�>      �?!   @c�>)���?Wp�=2a�Ϭ(�>8K�ߝ�>�������:              �?         <�       S�	q�z���A�*�
u
generator_loss_1*a	   ��I�?   ��I�?      �?!   ��I�?) c��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `�"�>   `�"�>      �?!   `�"�>)@®v�Y�=2�ѩ�-�>���%�>�������:              �?        �Ļ,�       �{�	̜����A*�
u
generator_loss_1*a	   @�>�?   @�>�?      �?!   @�>�?) �P~p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �:�>   �:�>      �?!   �:�>)��~�D��=2['�?��>K+�E���>�������:              �?        ���L�       ۞��	����A(*�
u
generator_loss_1*a	   �X�?   �X�?      �?!   �X�?) b�ԩ�n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��_�>   ��_�>      �?!   ��_�>) �aF��=2���%�>�uE����>�������:              �?        ~�4`�       ۞��	���AP*�
u
generator_loss_1*a	   �
ܰ?   �
ܰ?      �?!   �
ܰ?) 9����q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ৳�>   ৳�>      �?!   ৳�>) ��m�G�=2�iD*L��>E��a�W�>�������:              �?        G�o��       ۞��	l�8��Ax*�
u
generator_loss_1*a	   @dް?   @dް?      �?!   @dް?) !d���q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @A�>   @A�>      �?!   @A�>)�ŭ�'�=2�*��ڽ>�[�=�k�>�������:              �?        �9���       S�	���$��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) $ocm�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��Z�>   ��Z�>      �?!   ��Z�>) �8U��=2�XQ��>�����>�������:              �?        c�}��       S�	���0��A�*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?)@Z�,;;p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    -y�>    -y�>      �?!    -y�>) �����=2K+�E���>jqs&\��>�������:              �?        沱 �       S�	N��<��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ���m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �	�>   �	�>      �?!   �	�>) �`�X�=2E��a�W�>�ѩ�-�>�������:              �?        �D�E�       S�	e�OH��A�*�
u
generator_loss_1*a	   `H/�?   `H/�?      �?!   `H/�?)@b{z_p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��Q�>   ��Q�>      �?!   ��Q�>)�|Ҵ\{�=2
�/eq
�>;�"�q�>�������:              �?        �Y�;�       S�	2�"T��A�*�
u
generator_loss_1*a	    G��?    G��?      �?!    G��?) ��d�'o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �	�>    �	�>      �?!    �	�>) ���o�=2K+�E���>jqs&\��>�������:              �?        ?����       S�	���_��A�*�
u
generator_loss_1*a	    d�?    d�?      �?!    d�?) N����n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �,�>   �,�>      �?!   �,�>) 78A͈=2G&�$�>�*��ڽ>�������:              �?        �Ó�       S�	U��k��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) "�G1p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �
���=2K+�E���>jqs&\��>�������:              �?        �(�%�       S�	J��w��A�*�
u
generator_loss_1*a	    �6�?    �6�?      �?!    �6�?)@��@�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    d�>    d�>      �?!    d�>) ��^��|=2�u`P+d�>0�6�/n�>�������:              �?        *�@�       S�	u
_���A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) 1v3�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @�N�>   @�N�>      �?!   @�N�>) ��m�=2�����>
�/eq
�>�������:              �?        ͵8��       S�	�+D���A�*�
u
generator_loss_1*a	   �g��?   �g��?      �?!   �g��?) �]Wd�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ���:p=2����>豪}0ڰ>�������:              �?        D0_F�       S�	�:%���A�*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?)@��eu�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @\ѩ>   @\ѩ>      �?!   @\ѩ>)���y�d=2;9��R�>���?�ګ>�������:              �?        Z�f��       S�	������A�*�
u
generator_loss_1*a	   �@�?   �@�?      �?!   �@�?)�0��j?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)��!�C1�=2�ѩ�-�>���%�>�������:              �?        =x���       S�	\����A�*�
u
generator_loss_1*a	    |�?    |�?      �?!    |�?)  A#p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �&�>    �&�>      �?!    �&�>) ���S�=2['�?��>K+�E���>�������:              �?        �1��       S�	�޾��A�*�
u
generator_loss_1*a	    �ܰ?    �ܰ?      �?!    �ܰ?) � �q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    {�>    {�>      �?!    {�>) ��{�WO<2�i
�k>%���>�������:              �?        ����       S�	
�����A�*�
u
generator_loss_1*a	   �tɲ?   �tɲ?      �?!   �tɲ?) D`�.v?2��]$A�?�{ �ǳ�?�������:              �?        
w
discriminator_loss*a	   � �!>   � �!>      �?!   � �!>) 	��ǏS<2��-�z�!>4�e|�Z#>�������:              �?        ʿ�(�       S�	͂����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) gw<q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ���>>   ���>>      �?!   ���>>) _L����<2����W_>>p��Dp�@>�������:              �?        1�\��       S�	�����A�*�
u
generator_loss_1*a	   �(Z�?   �(Z�?      �?!   �(Z�?)@l@�M�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `��F>   `��F>      �?!   `��F>) ��mv�<2��Ő�;F>��8"uH>�������:              �?        ����       S�	������A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)@N~V_	q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �P>    �P>      �?!    �P>)@0C�R=�<2������M>28���FP>�������:              �?        {����       �{�	������A*�
u
generator_loss_1*a	   �_��?   �_��?      �?!   �_��?) թ�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @��j>   @��j>      �?!   @��j>)�l!���<2ڿ�ɓ�i>=�.^ol>�������:              �?        n	���       ۞��	u����A(*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@��V��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��	�>   ��	�>      �?!   ��	�>) DrLa$2=2���m!#�>�4[_>��>�������:              �?        �Z���       ۞��	{����AP*�
u
generator_loss_1*a	   �C��?   �C��?      �?!   �C��?) �]bYo?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �}(z>   �}(z>      �?!   �}(z>) 2�+�a=2E'�/��x>f^��`{>�������:              �?        `�C�       ۞��	�P���Ax*�
u
generator_loss_1*a	   �H,�?   �H,�?      �?!   �H,�?) B8Esl?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��	�>   ��	�>      �?!   ��	�>) ��Ri[=2u��6
�>T�L<�>�������:              �?        �=�G�       S�	�e�*��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ��c��h?2���g��?I���?�������:              �?        
w
discriminator_loss*a	    	?    	?      �?!    	?)@4��{x4>2��d�r?�5�i}1?�������:              �?        ��#��       S�	1��6��A�*�
u
generator_loss_1*a	    �?    �?      �?!    �?) ���Vl?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��	?   ��	?      �?!   ��	?) �Lw�$>2����?f�ʜ�7
?�������:              �?        J��       S�	��C��A�*�
u
generator_loss_1*a	    x��?    x��?      �?!    x��?)  �<m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �s��.>2��[�?1��a˲?�������:              �?        ��+�       S�	 nO��A�*�
u
generator_loss_1*a	   @�?   @�?      �?!   @�?)�x��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��*?   ��*?      �?!   ��*?) �Fv��>2��[�?1��a˲?�������:              �?        [��
�       S�	r�/[��A�*�
u
generator_loss_1*a	    �N�?    �N�?      �?!    �N�?)@�����p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `o�?   `o�?      �?!   `o�?) ���f(>2f�ʜ�7
?>h�'�?�������:              �?        ��.J�       S�	�Ag��A�*�
u
generator_loss_1*a	   `��?   `��?      �?!   `��?) �E<j?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   ��>?   ��>?      �?!   ��>?) 2NNc2'>2f�ʜ�7
?>h�'�?�������:              �?        �-F�       S�	�^s��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)���1�3k?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   ��:�>   ��:�>      �?!   ��:�>) dHּ��=2pz�w�7�>I��P=�>�������:              �?        u,���       S�	�_~��A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?) �E�� r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �?    �?      �?!    �?)  ��J,>2>h�'�?x?�x�?�������:              �?        W0{�       S�	P�����A�*�
u
generator_loss_1*a	   ��ۯ?   ��ۯ?      �?!   ��ۯ?) ���2�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ����#�=2pz�w�7�>I��P=�>�������:              �?        _�-3�       S�	�9���A�*�
u
generator_loss_1*a	   �G�?   �G�?      �?!   �G�?) ��ץl?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) aƐIH>21��a˲?6�]��?�������:              �?        )W�L�       S�	��F���A�*�
u
generator_loss_1*a	   `�>�?   `�>�?      �?!   `�>�?) �Q�g�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    \� ?    \� ?      �?!    \� ?)@p���_>2�FF�G ?��[�?�������:              �?        ��ۘ�       S�	ci���A�*�
u
generator_loss_1*a	   ��{�?   ��{�?      �?!   ��{�?) �e�V�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��Q�>   ��Q�>      �?!   ��Q�>) 2F�9��=2�h���`�>�ߊ4F��>�������:              �?        �����       S�	�����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) D��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    2-�>    2-�>      �?!    2-�>)  ��_>2>�?�s��>�FF�G ?�������:              �?        �d��       S�	My����A�*�
u
generator_loss_1*a	    "�?    "�?      �?!    "�?)  $�c�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) y+�[�>2�FF�G ?��[�?�������:              �?        �6l��       S�	Y���A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �T�,q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) �3��>2��Zr[v�>O�ʗ��>�������:              �?        
 ���       S�	y;U���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) d�� �q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ��2��=2})�l a�>pz�w�7�>�������:              �?        \ڳ��       S�	�2����A�*�
u
generator_loss_1*a	    �>�?    �>�?      �?!    �>�?) �����n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)� .��>2O�ʗ��>>�?�s��>�������:              �?        �x�B�       S�	������A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?)����x_k?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   �{m�>   �{m�>      �?!   �{m�>)���x�&>2I��P=�>��Zr[v�>�������:              �?        ��<m�       �{�	�N��A*�
u
generator_loss_1*a	    A��?    A��?      �?!    A��?) ��n�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) B�r�5�=28K�ߝ�>�h���`�>�������:              �?        %���       ۞��	Hx��A(*�
u
generator_loss_1*a	   @�(�?   @�(�?      �?!   @�(�?)���`Wn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @,��>   @,��>      �?!   @,��>)�0ճ���=28K�ߝ�>�h���`�>�������:              �?        �X�%�       ۞��	����AP*�
u
generator_loss_1*a	    �®?    �®?      �?!    �®?) F{��m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  U�u��=2�ѩ�-�>���%�>�������:              �?        �t��       ۞��	X�+��Ax*�
u
generator_loss_1*a	    �
�?    �
�?      �?!    �
�?) ��?�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��#�>   ��#�>      �?!   ��#�>) ��<wY�=2��~]�[�>��>M|K�>�������:              �?        H3U9�       S�	t�]7��A�*�
u
generator_loss_1*a	   @sZ�?   @sZ�?      �?!   @sZ�?)�����l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @"��>   @"��>      �?!   @"��>)��\��D�=2�iD*L��>E��a�W�>�������:              �?        {h��       S�	܋�C��A�*�
u
generator_loss_1*a	   ��b�?   ��b�?      �?!   ��b�?)�t��-i?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   �<p�>   �<p�>      �?!   �<p�>) Γ���>2��Zr[v�>O�ʗ��>�������:              �?        r��       S�	��%P��A�*�
u
generator_loss_1*a	   ��L�?   ��L�?      �?!   ��L�?) �G�K�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��(�>   ��(�>      �?!   ��(�>) $7���=2�f����>��(���>�������:              �?        ���       S�	��\��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) �=�n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) m�t���=2��(���>a�Ϭ(�>�������:              �?        ��,�       S�	uK(i��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)���j*�m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �}{�>   �}{�>      �?!   �}{�>)�(�|�)�=2E��a�W�>�ѩ�-�>�������:              �?        ]�N�       S�	�Ǖu��A�*�
u
generator_loss_1*a	   `>x�?   `>x�?      �?!   `>x�?)@*�h�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �+�>   �+�>      �?!   �+�>)@�WrE��=2���%�>�uE����>�������:              �?        X���       S�	�����A�*�
u
generator_loss_1*a	    ɭ?    ɭ?      �?!    ɭ?)  ��n�k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �$b%%�=2})�l a�>pz�w�7�>�������:              �?        }�Ε�       S�	o�����A�*�
u
generator_loss_1*a	    �V�?    �V�?      �?!    �V�?) ��4�n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @`�>   @`�>      �?!   @`�>) ����=2�uE����>�f����>�������:              �?        r�_i�       S�	 T���A�*�
u
generator_loss_1*a	   ��w�?   ��w�?      �?!   ��w�?)@.ָ��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) H�����=2�iD*L��>E��a�W�>�������:              �?        L%V�       S�	sT����A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?)���j�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    2��>    2��>      �?!    2��>) @��|�=2�ѩ�-�>���%�>�������:              �?        0��       S�	$����A�*�
u
generator_loss_1*a	   ��3�?   ��3�?      �?!   ��3�?) U^Z"�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �qe�>   �qe�>      �?!   �qe�>)@�= �=2��~]�[�>��>M|K�>�������:              �?        ���M�       S�	�����A�*�
u
generator_loss_1*a	    J��?    J��?      �?!    J��?)  _��o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)��`�C��=2E��a�W�>�ѩ�-�>�������:              �?        xCT �       S�	�1���A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) A|�'q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �\�>    �\�>      �?!    �\�>) ��[,�=2���%�>�uE����>�������:              �?        �ܖ��       S�	������A�*�
u
generator_loss_1*a	   �H<�?   �H<�?      �?!   �H<�?)�du�>}n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �"?    �"?      �?!    �"?)@@���Y>2�FF�G ?��[�?�������:              �?        �=��       S�	U
g���A�*�
u
generator_loss_1*a	   `.D�?   `.D�?      �?!   `.D�?)@j�J�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �k�>    �k�>      �?!    �k�>) ���_��=2pz�w�7�>I��P=�>�������:              �?        ̺�5�       S�	������A�*�
u
generator_loss_1*a	   �t��?   �t��?      �?!   �t��?)��e���m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �i��>   �i��>      �?!   �i��>)�x�a��=2��(���>a�Ϭ(�>�������:              �?        �wo�       S�	�����A�*�
u
generator_loss_1*a	   @+ï?   @+ï?      �?!   @+ï?)�tV#ʆo?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@^%3���=2�ߊ4F��>})�l a�>�������:              �?        �����       S�	�:��A�*�
u
generator_loss_1*a	   `z��?   `z��?      �?!   `z��?) �LTk?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) 2���>2I��P=�>��Zr[v�>�������:              �?        w}��       �{�	�����A*�
u
generator_loss_1*a	   ��Ư?   ��Ư?      �?!   ��Ư?) ���o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �>��>   �>��>      �?!   �>��>)@�ϥ��=2�uE����>�f����>�������:              �?        |��       ۞��	���%��A(*�
u
generator_loss_1*a	    ,֭?    ,֭?      �?!    ,֭?) ��,��k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) @�wl	t=2豪}0ڰ>��n����>�������:              �?        	7�       ۞��	w�72��AP*�
u
generator_loss_1*a	   �-A�?   �-A�?      �?!   �-A�?) ğ�͆n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    f�>    f�>      �?!    f�>) ��_ǅ=25�"�g��>G&�$�>�������:              �?        ժ��       ۞��	�
�>��Ax*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?) �Cc#n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    1��>    1��>      �?!    1��>)@Ԃm�B�=2��>M|K�>�_�T�l�>�������:              �?        �b�
�       S�	��K��A�*�
u
generator_loss_1*a	   �6@�?   �6@�?      �?!   �6@�?)@~��n�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `)�>   `)�>      �?!   `)�>) �m�=28K�ߝ�>�h���`�>�������:              �?        k?�       S�	(iX��A�*�
u
generator_loss_1*a	    ]��?    ]��?      �?!    ]��?) ��To?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ^�
?    ^�
?      �?!    ^�
?)  �
��%>2f�ʜ�7
?>h�'�?�������:              �?        ���N�       S�	*�)e��A�*�
u
generator_loss_1*a	   ��)�?   ��)�?      �?!   ��)�?)�����Xn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �	?    �	?      �?!    �	?)@|��$2>2x?�x�?��d�r?�������:              �?        h#Z��       S�	�zr��A�*�
u
generator_loss_1*a	   �=��?   �=��?      �?!   �=��?) Q�k�Hq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��>?   ��>?      �?!   ��>?)�4�p�� >26�]��?����?�������:              �?        �����       S�	�1�~��A�*�
u
generator_loss_1*a	    ᔮ?    ᔮ?      �?!    ᔮ?) n��9m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�T'�=2})�l a�>pz�w�7�>�������:              �?        rr���       S�	P�����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)���F*n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �_�$?   �_�$?      �?!   �_�$?) �ѼuZ>2U�4@@�$?+A�F�&?�������:              �?        ھ�*�       S�	E_���A�*�
u
generator_loss_1*a	   �ү?   �ү?      �?!   �ү?)�0Dgi�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) �<3�>2��[�?1��a˲?�������:              �?        H���       S�	g����A�*�
u
generator_loss_1*a	   ��4�?   ��4�?      �?!   ��4�?)�|�qnn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �\b�>   �\b�>      �?!   �\b�>) ��+o>2O�ʗ��>>�?�s��>�������:              �?        /��m�       S�	g���A�*�
u
generator_loss_1*a	   �M��?   �M��?      �?!   �M��?)��m�k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�к&�>2O�ʗ��>>�?�s��>�������:              �?        /��D�       S�	Ƹ����A�*�
u
generator_loss_1*a	   �O@�?   �O@�?      �?!   �O@�?)�����l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �7?    �7?      �?!    �7?) ��y�SB>2�vV�R9?��ڋ?�������:              �?        T5��       S�	j�j���A�*�
u
generator_loss_1*a	    0į?    0į?      �?!    0į?) `�Јo?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @�}
?   @�}
?      �?!   @�}
?)�\nՃ�%>2f�ʜ�7
?>h�'�?�������:              �?        7��       S�	��:���A�*�
u
generator_loss_1*a	    p��?    p��?      �?!    p��?) �v�}8j?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   �T?   �T?      �?!   �T?) �)&T�(>2f�ʜ�7
?>h�'�?�������:              �?        �����       S�	TY���A�*�
u
generator_loss_1*a	    7��?    7��?      �?!    7��?) �l�i?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   �y?   �y?      �?!   �y?) D��>2��[�?1��a˲?�������:              �?        |8���       S�	z����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)��G��zj?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   �@�?   �@�?      �?!   �@�?)@s?k>21��a˲?6�]��?�������:              �?        �jf��       S�	Wx����A�*�
u
generator_loss_1*a	    R �?    R �?      �?!    R �?)  ҄�Fn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �]D�>   �]D�>      �?!   �]D�>) 2q�{�
>2O�ʗ��>>�?�s��>�������:              �?        �� %�       S�	`����A�*�
u
generator_loss_1*a	   �!ϭ?   �!ϭ?      �?!   �!ϭ?) �q��k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) i;t�S>2>�?�s��>�FF�G ?�������:              �?        z~���       S�	�~���A�*�
u
generator_loss_1*a	    n�?    n�?      �?!    n�?) n����n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)�t"��>2O�ʗ��>>�?�s��>�������:              �?        ��]�       S�	��g%��A�*�
u
generator_loss_1*a	   �yl�?   �yl�?      �?!   �yl�?) E�\��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �]e�>   �]e�>      �?!   �]e�>) d�����=2�uE����>�f����>�������:              �?        �d�5�       �{�	��+2��A*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?)��Po?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �ɭ�>   �ɭ�>      �?!   �ɭ�>) ��\���=2���%�>�uE����>�������:              �?        ���L�       ۞��	�zA?��A(*�
u
generator_loss_1*a	    �ǯ?    �ǯ?      �?!    �ǯ?)  Ѭo�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��T�>   ��T�>      �?!   ��T�>) ���y+�=2I��P=�>��Zr[v�>�������:              �?        )����       ۞��	��5L��AP*�
u
generator_loss_1*a	    Q�?    Q�?      �?!    Q�?)  Yܥn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) '�j�>2>�?�s��>�FF�G ?�������:              �?        ����       ۞��	��'Y��Ax*�
u
generator_loss_1*a	   @M]�?   @M]�?      �?!   @M]�?)�|��սn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @Yn�>   @Yn�>      �?!   @Yn�>) �!U���=2pz�w�7�>I��P=�>�������:              �?        XE�       S�	x�"f��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ѝ\�q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `0��>   `0��>      �?!   `0��>) !��k�=2['�?��>K+�E���>�������:              �?        D�j�       S�	�>s��A�*�
u
generator_loss_1*a	   `T�?   `T�?      �?!   `T�?) 1�嬫n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �v��>   �v��>      �?!   �v��>) Yyq�N�=2K+�E���>jqs&\��>�������:              �?        Y���       S�	4:,���A�*�
u
generator_loss_1*a	   �v�?   �v�?      �?!   �v�?) �NN� o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    r��>    r��>      �?!    r��>)  � K��=2��(���>a�Ϭ(�>�������:              �?        �C��       S�	?�3���A�*�
u
generator_loss_1*a	   �Z'�?   �Z'�?      �?!   �Z'�?) �/�Op?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ^y�>    ^y�>      �?!    ^y�>)  ��0V�=2;�"�q�>['�?��>�������:              �?        �E3�       S�	�>���A�*�
u
generator_loss_1*a	    ߭?    ߭?      �?!    ߭?)  �k�k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @=��>   @=��>      �?!   @=��>)�<E�
�=2['�?��>K+�E���>�������:              �?        K����       S�	e�^���A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) �7�U�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�d���c�=2�iD*L��>E��a�W�>�������:              �?        �s���       S�	8�q���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ���i?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) dE��7�=2�h���`�>�ߊ4F��>�������:              �?        ��3�       S�	�l����A�*�
u
generator_loss_1*a	    Ʈ?    Ʈ?      �?!    Ʈ?)  ʾ+�m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �h�>    �h�>      �?!    �h�>) v-���=2��~���>�XQ��>�������:              �?        �5"<�       S�	c�����A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?)��k�Fuj?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   ��]�>   ��]�>      �?!   ��]�>) ��辎=2�*��ڽ>�[�=�k�>�������:              �?        ��L�       S�	������A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?)  1��q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �]�>   �]�>      �?!   �]�>)��R�=20�6�/n�>5�"�g��>�������:              �?        4��       S�	@�����A�*�
u
generator_loss_1*a	   @+U�?   @+U�?      �?!   @+U�?) ����p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) a��U�=2�_�T�l�>�iD*L��>�������:              �?        �#�       S�	�L	���A�*�
u
generator_loss_1*a	    Ta�?    Ta�?      �?!    Ta�?)  9��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ��M�>   ��M�>      �?!   ��M�>)@f���=2�f����>��(���>�������:              �?        |���       S�	��1��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �ثYIm?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �~I�>   �~I�>      �?!   �~I�>) $8���=2�XQ��>�����>�������:              �?        �a��       S�	�KK��A�*�
u
generator_loss_1*a	   ��,�?   ��,�?      �?!   ��,�?) �d|tl?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `m��>   `m��>      �?!   `m��>)@�S��!�=2K+�E���>jqs&\��>�������:              �?        ���       S�	�/j��A�*�
u
generator_loss_1*a	   @�?   @�?      �?!   @�?) �-4�oq?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    �R�>    �R�>      �?!    �R�>) (G�	�=25�"�g��>G&�$�>�������:              �?        c���       S�	sB�*��A�*�
u
generator_loss_1*a	   �Y�?   �Y�?      �?!   �Y�?)�����@n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �=ڸ>   �=ڸ>      �?!   �=ڸ>) ��,M�=25�"�g��>G&�$�>�������:              �?        d)>I�       S�	#��7��A�*�
u
generator_loss_1*a	   @1V�?   @1V�?      �?!   @1V�?)����n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  ��}�=2
�/eq
�>;�"�q�>�������:              �?        �[u�       S�	(E��A�*�
u
generator_loss_1*a	    ӄ�?    ӄ�?      �?!    ӄ�?) H/��:k?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@609�=�=2��~]�[�>��>M|K�>�������:              �?        #%^#�       �{�	)�@R��A*�
u
generator_loss_1*a	   ��d�?   ��d�?      �?!   ��d�?)���z��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �CV�>   �CV�>      �?!   �CV�>)�p���¬=2['�?��>K+�E���>�������:              �?        2;��       ۞��	�2�_��A(*�
u
generator_loss_1*a	   �R��?   �R��?      �?!   �R��?) W���m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �2u�=2['�?��>K+�E���>�������:              �?        � �-�       ۞��	mym��AP*�
u
generator_loss_1*a	   �u=�?   �u=�?      �?!   �u=�?) ]"���l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ��;��=2�uE����>�f����>�������:              �?        �"Ph�       ۞��	x2Yz��Ax*�
u
generator_loss_1*a	   �d��?   �d��?      �?!   �d��?) �;M�2k?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   @/9�>   @/9�>      �?!   @/9�>)�ę�R}�=2
�/eq
�>;�"�q�>�������:              �?        %4Cj�       S�	i����A�*�
u
generator_loss_1*a	    �t�?    �t�?      �?!    �t�?)  @��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ?�>    ?�>      �?!    ?�>) T��=2
�/eq
�>;�"�q�>�������:              �?        �XY��       S�	�T���A�*�
u
generator_loss_1*a	   `���?   `���?      �?!   `���?) �	��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    S�>    S�>      �?!    S�>) �M�姰=2K+�E���>jqs&\��>�������:              �?        �F\��       S�	E7���A�*�
u
generator_loss_1*a	   �p&�?   �p&�?      �?!   �p&�?)@��=Mp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @Σ�>   @Σ�>      �?!   @Σ�>) ����=2��~���>�XQ��>�������:              �?        �)�N�       S�	4����A�*�
u
generator_loss_1*a	    �?    �?      �?!    �?) H����k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) �'i���=2�XQ��>�����>�������:              �?        [�m�       S�	�c����A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) `\�m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �ʹ�>   �ʹ�>      �?!   �ʹ�>)@��|�=2�[�=�k�>��~���>�������:              �?        &�wv�       S�	<Y}���A�*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?)�`\}��k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �7�>   �7�>      �?!   �7�>) ��ļt=2豪}0ڰ>��n����>�������:              �?        @����       S�	�����A�*�
u
generator_loss_1*a	   ��m�?   ��m�?      �?!   ��m�?) !�ͩ�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@���=2�XQ��>�����>�������:              �?        �`�       S�	�1+���A�*�
u
generator_loss_1*a	   �$n�?   �$n�?      �?!   �$n�?) D36?�p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @|�>   @|�>      �?!   @|�>)�p�}�=2�*��ڽ>�[�=�k�>�������:              �?        b%��       S�	������A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?)@���;p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �ī�>   �ī�>      �?!   �ī�>)��}o���=25�"�g��>G&�$�>�������:              �?        �;{��       S�	� ����A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) ��@;o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �vƳ>   �vƳ>      �?!   �vƳ>) �M(�px=2��n����>�u`P+d�>�������:              �?        �F�4�       S�	��K��A�*�
u
generator_loss_1*a	   �4�?   �4�?      �?!   �4�?) ג�8mn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��4�>   ��4�>      �?!   ��4�>) �'��u�=25�"�g��>G&�$�>�������:              �?        �GZ�       S�	S����A�*�
u
generator_loss_1*a	    M�?    M�?      �?!    M�?) H�a�j?2���g��?I���?�������:              �?        
w
discriminator_loss*a	    8��>    8��>      �?!    8��>) pP�y��=2�����>
�/eq
�>�������:              �?        �o�*�       S�	�~6(��A�*�
u
generator_loss_1*a	   @�ȯ?   @�ȯ?      �?!   @�ȯ?)�0�K7�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) '[�]�=25�"�g��>G&�$�>�������:              �?        �����       S�	�E�5��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) � ���o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @�ն>   @�ն>      �?!   @�ն>)�h�}:K�=20�6�/n�>5�"�g��>�������:              �?        a���       S�	�zKC��A�*�
u
generator_loss_1*a	   �.�?   �.�?      �?!   �.�?)@�w�fp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �r�>   �r�>      �?!   �r�>)�|��-�=20�6�/n�>5�"�g��>�������:              �?        �:���       S�	���P��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �=J`o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)  �v��=2G&�$�>�*��ڽ>�������:              �?        ��?k�       S�	H�3^��A�*�
u
generator_loss_1*a	    �˰?    �˰?      �?!    �˰?)@�����q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �!��>   �!��>      �?!   �!��>)@��5w��=2�[�=�k�>��~���>�������:              �?        ��sv�       S�	u�k��A�*�
u
generator_loss_1*a	   �-��?   �-��?      �?!   �-��?) �(�=?k?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�T4��=2�*��ڽ>�[�=�k�>�������:              �?        ;�}��       �{�	x6y��A*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?)�,3s;To?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �uG��=2�[�=�k�>��~���>�������:              �?        Mm7�       ۞��	�4ˆ��A(*�
u
generator_loss_1*a	   @��?   @��?      �?!   @��?)� �<q?l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��f�>   ��f�>      �?!   ��f�>) �y|���=2��>M|K�>�_�T�l�>�������:              �?        ph�C�       ۞��	u�j���AP*�
u
generator_loss_1*a	   `���?   `���?      �?!   `���?) _5k?2���g��?I���?�������:              �?        
w
discriminator_loss*a	    �>    �>      �?!    �>)  FU��=2jqs&\��>��~]�[�>�������:              �?        �v"��       ۞��	ˮ���Ax*�
u
generator_loss_1*a	    d�?    d�?      �?!    d�?)@�.�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    "��>    "��>      �?!    "��>) @�tG�s=2豪}0ڰ>��n����>�������:              �?        <&���       S�	q�����A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) rW��o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) ���=2jqs&\��>��~]�[�>�������:              �?        �U��       S�	y�Z���A�*�
u
generator_loss_1*a	   @���?   @���?      �?!   @���?)��u"2�m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@ [�l~�=2�XQ��>�����>�������:              �?        �#��       S�	� ���A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ���n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �x�>   �x�>      �?!   �x�>) ¥��{�=2['�?��>K+�E���>�������:              �?        �ٴ�       S�	<,����A�*�
u
generator_loss_1*a	   �d��?   �d��?      �?!   �d��?) k {yo?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �xm�>   �xm�>      �?!   �xm�>) Iǘ���=2K+�E���>jqs&\��>�������:              �?        ����       S�	M�V���A�*�
u
generator_loss_1*a	   �1��?   �1��?      �?!   �1��?) �����k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��^�>   ��^�>      �?!   ��^�>)@�H"���=2�f����>��(���>�������:              �?         ����       S�	�����A�*�
u
generator_loss_1*a	   ��d�?   ��d�?      �?!   ��d�?) �˖��j?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ���h�=2�iD*L��>E��a�W�>�������:              �?        ���~�       S�	��� ��A�*�
u
generator_loss_1*a	    H�?    H�?      �?!    H�?) ȃѶ�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �ͻЖ�=2�f����>��(���>�������:              �?        �B ��       S�	�h ��A�*�
u
generator_loss_1*a	   �3��?   �3��?      �?!   �3��?) Iö4@m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �e�>   �e�>      �?!   �e�>)@0Ң~��=2��>M|K�>�_�T�l�>�������:              �?        LU�q�       S�	�" ��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)����G9o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��I�>   ��I�>      �?!   ��I�>) �<7�@�=2��~���>�XQ��>�������:              �?        ��ɣ�       S�	G��* ��A�*�
u
generator_loss_1*a	    ~X�?    ~X�?      �?!    ~X�?)@8B��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   �'��>   �'��>      �?!   �'��>) �q��=2��~���>�XQ��>�������:              �?        ; ���       S�	l�8 ��A�*�
u
generator_loss_1*a	   @{��?   @{��?      �?!   @{��?) i%�`q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �N;��=2�*��ڽ>�[�=�k�>�������:              �?        ��#��       S�	��FF ��A�*�
u
generator_loss_1*a	   @M��?   @M��?      �?!   @M��?)�|�k�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �
��>   �
��>      �?!   �
��>)@�Y���=2�XQ��>�����>�������:              �?        �T��       S�	��T ��A�*�
u
generator_loss_1*a	   @�!�?   @�!�?      �?!   @�!�?) �7�Cp?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    X��>    X��>      �?!    X��>)@`���w=2��n����>�u`P+d�>�������:              �?        �({C�       S�	�x�a ��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) ���.n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��uݳ�=2G&�$�>�*��ڽ>�������:              �?        4x�k�       S�	��o ��A�*�
u
generator_loss_1*a	   �9�?   �9�?      �?!   �9�?)@X�"�:p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��h�>   ��h�>      �?!   ��h�>) ^uz�=2G&�$�>�*��ڽ>�������:              �?        ݑ[��       S�	��} ��A�*�
u
generator_loss_1*a	   ����?   ����?      �?!   ����?) �a"sRm?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �%�>   �%�>      �?!   �%�>) ܈MS]�=25�"�g��>G&�$�>�������:              �?        �%�w�       S�	{�P� ��A�*�
u
generator_loss_1*a	   `?��?   `?��?      �?!   `?��?) �u2io?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��V�>   ��V�>      �?!   ��V�>) �\���=2�*��ڽ>�[�=�k�>�������:              �?        m�
�       S�	T�'� ��A�*�
u
generator_loss_1*a	   @>�?   @>�?      �?!   @>�?)�|]+єl?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    9��>    9��>      �?!    9��>) �� r�=2�iD*L��>E��a�W�>�������:              �?        a	��       �{�	e<� ��A*�
u
generator_loss_1*a	   `a�?   `a�?      �?!   `a�?) O8=O7n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)��hS���=2�ѩ�-�>���%�>�������:              �?        ��Ƀ�       ۞��	Lc�� ��A(*�
u
generator_loss_1*a	    ,D�?    ,D�?      �?!    ,D�?)  yvz�p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �-��>   �-��>      �?!   �-��>)@J]�=2jqs&\��>��~]�[�>�������:              �?        L�%I�       ۞��	)��� ��AP*�
u
generator_loss_1*a	   @X�?   @X�?      �?!   @X�?)�@h��l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @�i�>   @�i�>      �?!   @�i�>) ��"60�=2��~���>�XQ��>�������:              �?        P[�6�       ۞��	
�� ��Ax*�
u
generator_loss_1*a	   ��H�?   ��H�?      �?!   ��H�?) ��k�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �C�>   �C�>      �?!   �C�>)@
�گ��=2�[�=�k�>��~���>�������:              �?        e����       S�	`��� ��A�*�
u
generator_loss_1*a	   @C��?   @C��?      �?!   @C��?)�T ^lm?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)��'����=2G&�$�>�*��ڽ>�������:              �?        �����       S�	(ʩ� ��A�*�
u
generator_loss_1*a	   �l�?   �l�?      �?!   �l�?)  Z��l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) ����=20�6�/n�>5�"�g��>�������:              �?        ����       S�	R��� ��A�*�
u
generator_loss_1*a	    �*�?    �*�?      �?!    �*�?) lr���h?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   @R��>   @R��>      �?!   @R��>) �:�z=2�u`P+d�>0�6�/n�>�������:              �?        h�eQ�       S�	�!��A�*�
u
generator_loss_1*a	    Ys�?    Ys�?      �?!    Ys�?) O���p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) 1w�)$~=2�u`P+d�>0�6�/n�>�������:              �?        cv���       S�	�X�!��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) �*,��o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �U0�>   �U0�>      �?!   �U0�>)@��hyy=2��n����>�u`P+d�>�������:              �?        {<���       S�	1�$!��A�*�
u
generator_loss_1*a	    tƯ?    tƯ?      �?!    tƯ?) �$}O�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @'��>   @'��>      �?!   @'��>)�$���~h=2���?�ګ>����>�������:              �?        ~.�R�       S�	'�2!��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)  ��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �Cr�>   �Cr�>      �?!   �Cr�>)�pGWj��=2;�"�q�>['�?��>�������:              �?        W%Hx�       S�	]�@!��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) 2�`L�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �4.�>   �4.�>      �?!   �4.�>) ����r�=2�[�=�k�>��~���>�������:              �?        �����       S�	�O!��A�*�
u
generator_loss_1*a	   ��9�?   ��9�?      �?!   ��9�?)���G�xn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @�)�>   @�)�>      �?!   @�)�>)����{Ɉ=2G&�$�>�*��ڽ>�������:              �?        �       S�	�6H]!��A�*�
u
generator_loss_1*a	   �-�?   �-�?      �?!   �-�?) �(An�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �>    �>      �?!    �>) ����~=2�u`P+d�>0�6�/n�>�������:              �?        ��i�       S�	W�Vk!��A�*�
u
generator_loss_1*a	    ���?    ���?      �?!    ���?) �>6�o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �ܴ>    �ܴ>      �?!    �ܴ>) @bi.3{=2�u`P+d�>0�6�/n�>�������:              �?        lL���       S�	��ly!��A�*�
u
generator_loss_1*a	   ��,�?   ��,�?      �?!   ��,�?)����sl?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �fx�>   �fx�>      �?!   �fx�>) RZ�6�=20�6�/n�>5�"�g��>�������:              �?        A���       S�	)>��!��A�*�
u
generator_loss_1*a	    +`�?    +`�?      �?!    +`�?)@<�E��p?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    _�>    _�>      �?!    _�>)@��my=2��n����>�u`P+d�>�������:              �?        ��R��       S�	�l��!��A�*�
u
generator_loss_1*a	   @?��?   @?��?      �?!   @?��?)����m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��,�>   ��,�>      �?!   ��,�>)�hqnΈ=2G&�$�>�*��ڽ>�������:              �?        ˸�]�       S�	��أ!��A�*�
u
generator_loss_1*a	   `�\�?   `�\�?      �?!   `�\�?) G�Gm�n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �컵>   �컵>      �?!   �컵>) )���}=2�u`P+d�>0�6�/n�>�������:              �?        �r�4�       S�	�
�!��A�*�
u
generator_loss_1*a	   �xu�?   �xu�?      �?!   �xu�?) ���H�n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) ���aI{=2�u`P+d�>0�6�/n�>�������:              �?        ���7�       S�	>�!��A�*�
u
generator_loss_1*a	   �l{�?   �l{�?      �?!   �l{�?)��u���n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �|Әv=2��n����>�u`P+d�>�������:              �?        �c��       S�	8O�!��A�*�
u
generator_loss_1*a	   �ƪ�?   �ƪ�?      �?!   �ƪ�?) ����s?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    G��>    G��>      �?!    G��>) ��Z4k=2���?�ګ>����>�������:              �?        �#P�       �{�	]�n�!��A*�
u
generator_loss_1*a	   @�Z�?   @�Z�?      �?!   @�Z�?)�=
�j?2���g��?I���?�������:              �?        
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@���F�=2�[�=�k�>��~���>�������:              �?        $9��       ۞��	9<��!��A(*�
u
generator_loss_1*a	   ��i�?   ��i�?      �?!   ��i�?) .0p	k?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   @�Ұ>   @�Ұ>      �?!   @�Ұ>) �6P�q=2����>豪}0ڰ>�������:              �?        �wb��       ۞��	�%�!��AP*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)��g8Om?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��r�>   ��r�>      �?!   ��r�>)@b�W)�p=2����>豪}0ڰ>�������:              �?        lw��       ۞��	�r"��Ax*�
u
generator_loss_1*a	   �(
�?   �(
�?      �?!   �(
�?)��-�n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �u\�>   �u\�>      �?!   �u\�>) 4zo�"i=2���?�ګ>����>�������:              �?        T��C�       S�	T(�"��A�*�
u
generator_loss_1*a	    9�?    9�?      �?!    9�?) ���Lj?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   �\�>   �\�>      �?!   �\�>) ��� Ly=2��n����>�u`P+d�>�������:              �?        �Z�E�       S�	�L�#"��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)�@�\�n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @�
�>   @�
�>      �?!   @�
�>)� l[x�c=2�5�L�>;9��R�>�������:              �?        )�HK�       S�	t!%2"��A�*�
u
generator_loss_1*a	   @(�?   @(�?      �?!   @(�?)��nUt l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    ^��>    ^��>      �?!    ^��>)  �<�o=2����>豪}0ڰ>�������:              �?        1NRT�       S�	��T@"��A�*�
u
generator_loss_1*a	   ��?   ��?      �?!   ��?) �\�YZl?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �\��>   �\��>      �?!   �\��>) j�[Wd=2;9��R�>���?�ګ>�������:              �?        ��0g�       S�	���N"��A�*�
u
generator_loss_1*a	   ��ٮ?   ��ٮ?      �?!   ��ٮ?) �?�+�m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    V�>    V�>      �?!    V�>)  G�;�d=2;9��R�>���?�ګ>�������:              �?        zKhj�       S�	��\"��A�*�
u
generator_loss_1*a	   �	��?   �	��?      �?!   �	��?) �j۲m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �tp=2����>豪}0ڰ>�������:              �?        K�v��       S�	�L4k"��A�*�
u
generator_loss_1*a	   ��~�?   ��~�?      �?!   ��~�?) ��K<�n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) � /}S=2�MZ��K�>��|�~�>�������:              �?        ����       S�	�twy"��A�*�
u
generator_loss_1*a	   �ޟ�?   �ޟ�?      �?!   �ޟ�?)���[�Nm?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `fʠ>   `fʠ>      �?!   `fʠ>)@
�͞Q=2�u��gr�>�MZ��K�>�������:              �?        5�g��       S�	��ʇ"��A�*�
u
generator_loss_1*a	   �"/�?   �"/�?      �?!   �"/�?) w���h?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�d)*Ua=2�5�L�>;9��R�>�������:              �?        �;��       S�	�5�"��A�*�
u
generator_loss_1*a	   ��կ?   ��կ?      �?!   ��կ?)������o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �j�>   �j�>      �?!   �j�>)@̩e1U=2�MZ��K�>��|�~�>�������:              �?        Y�m��       S�	�Q��"��A�*�
u
generator_loss_1*a	   @�Z�?   @�Z�?      �?!   @�Z�?)��Z((�n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    �R�>    �R�>      �?!    �R�>) ���Y=2��|�~�>���]���>�������:              �?        :�F�       S�	y��"��A�*�
u
generator_loss_1*a	   @�>�?   @�>�?      �?!   @�>�?) 9��{�r?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�8�ڗ�A=2.��fc��>39W$:��>�������:              �?        �����       S�	.�u�"��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@�m���q?2����iH�?��]$A�?�������:              �?        
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  �nޢ=2
�/eq
�>;�"�q�>�������:              �?        E���       S�	a ��"��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?) Ƚ���k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    r�>    r�>      �?!    r�>)@j"?�=2K+�E���>jqs&\��>�������:              �?        �ݛ��       S�	l?^�"��A�*�
u
generator_loss_1*a	   ಉ�?   ಉ�?      �?!   ಉ�?) ⅏Isi?2���g��?I���?�������:              �?        
w
discriminator_loss*a	   �iZ�>   �iZ�>      �?!   �iZ�>)�x5x�
�=2��(���>a�Ϭ(�>�������:              �?        �+�Z�       S�	����"��A�*�
u
generator_loss_1*a	   �s�?   �s�?      �?!   �s�?) q�=j�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)�|-�=�=2�*��ڽ>�[�=�k�>�������:              �?        �"��       S�	�W9�"��A�*�
u
generator_loss_1*a	   �y�?   �y�?      �?!   �y�?)�8�_�'n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �U�>   �U�>      �?!   �U�>) �X�і=2��~���>�XQ��>�������:              �?        �,_��       S�	*�	#��A�*�
u
generator_loss_1*a	    ´�?    ´�?      �?!    ´�?)  ���k?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �F�>   �F�>      �?!   �F�>) R��ʍ=2�*��ڽ>�[�=�k�>�������:              �?        ����       �{�	s�B#��A*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?)@~�@�7p?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @&�>   @&�>      �?!   @&�>) �aF�L�=2�*��ڽ>�[�=�k�>�������:              �?        ��l��       ۞��	њA*#��A(*�
u
generator_loss_1*a	   `M�?   `M�?      �?!   `M�?) ��!�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) CG�s�=2;�"�q�>['�?��>�������:              �?        f���       ۞��	P�t;#��AP*�
u
generator_loss_1*a	   �䞯?   �䞯?      �?!   �䞯?) �Ϯ�>o?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) y�ر��=2�����>
�/eq
�>�������:              �?        `���       ۞��	��K#��Ax*�
u
generator_loss_1*a	    IA�?    IA�?      �?!    IA�?) �6��n?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �A¼>   �A¼>      �?!   �A¼>)��M�؉=2G&�$�>�*��ڽ>�������:              �?        @2H��       S�	��G[#��A�*�
u
generator_loss_1*a	   @$��?   @$��?      �?!   @$��?)�o�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `<��>   `<��>      �?!   `<��>)@�7�$O�=2��~]�[�>��>M|K�>�������:              �?        �'N��       S�	,�k#��A�*�
u
generator_loss_1*a	   �i.�?   �i.�?      �?!   �i.�?)�xe�Iwl?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �i��>   �i��>      �?!   �i��>)�x���=2['�?��>K+�E���>�������:              �?        �-�       S�	�]z#��A�*�
u
generator_loss_1*a	    ��?    ��?      �?!    ��?)  �Ψn?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) S�
��=2E��a�W�>�ѩ�-�>�������:              �?        ^��       S�	w���#��A�*�
u
generator_loss_1*a	   ���?   ���?      �?!   ���?) "\�Zo?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	    *��>    *��>      �?!    *��>) @�Cr�=2jqs&\��>��~]�[�>�������:              �?        F�ɑ�       S�	;�#��A�*�
u
generator_loss_1*a	    /��?    /��?      �?!    /��?) f�^�l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   @
��>   @
��>      �?!   @
��>)�H"D�=2['�?��>K+�E���>�������:              �?        �FDk�       S�	nu �#��A�*�
u
generator_loss_1*a	   @UO�?   @UO�?      �?!   @UO�?)�υ��l?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	   �^@�>   �^@�>      �?!   �^@�>) $.�9�=2��~]�[�>��>M|K�>�������:              �?        ߨ�y�       S�	^C��#��A�*�
u
generator_loss_1*a	   � ��?   � ��?      �?!   � ��?)��=�z�m?2I���?����iH�?�������:              �?        
w
discriminator_loss*a	     ^�>     ^�>      �?!     ^�>)   @��=2��~���>�XQ��>�������:              �?        �&o