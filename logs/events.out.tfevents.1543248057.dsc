       ЃK"	  @.џжAbrain.Event:20foЦ     ы­Yк	`g.џжA"

Encoder/real_inPlaceholder*
dtype0*/
_output_shapes
:џџџџџџџџџ*$
shape:џџџџџџџџџ
f
Encoder/Reshape/shapeConst*
valueB"џџџџ  *
dtype0*
_output_shapes
:

Encoder/ReshapeReshapeEncoder/real_inEncoder/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
л
KEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
Э
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *HYН*
dtype0*
_output_shapes
: 
Э
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
С
SEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformKEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ц
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
_output_shapes
: 
к
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulSEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:

Ь
EEncoder/first_layer/fully_connected/kernel/Initializer/random_uniformAddIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:

с
*Encoder/first_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container *
shape:

С
1Encoder/first_layer/fully_connected/kernel/AssignAssign*Encoder/first_layer/fully_connected/kernelEEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

б
/Encoder/first_layer/fully_connected/kernel/readIdentity*Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel
Ц
:Encoder/first_layer/fully_connected/bias/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
г
(Encoder/first_layer/fully_connected/bias
VariableV2*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ћ
/Encoder/first_layer/fully_connected/bias/AssignAssign(Encoder/first_layer/fully_connected/bias:Encoder/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ц
-Encoder/first_layer/fully_connected/bias/readIdentity(Encoder/first_layer/fully_connected/bias*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:
Я
*Encoder/first_layer/fully_connected/MatMulMatMulEncoder/Reshape/Encoder/first_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
л
+Encoder/first_layer/fully_connected/BiasAddBiasAdd*Encoder/first_layer/fully_connected/MatMul-Encoder/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
i
$Encoder/first_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>
Џ
"Encoder/first_layer/leaky_relu/mulMul$Encoder/first_layer/leaky_relu/alpha+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
­
Encoder/first_layer/leaky_reluMaximum"Encoder/first_layer/leaky_relu/mul+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
н
LEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      
Я
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *qФН
Я
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *qФ=*
dtype0*
_output_shapes
: 
Ф
TEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
Ъ
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
_output_shapes
: 
о
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
а
FEncoder/second_layer/fully_connected/kernel/Initializer/random_uniformAddJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
у
+Encoder/second_layer/fully_connected/kernel
VariableV2*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Х
2Encoder/second_layer/fully_connected/kernel/AssignAssign+Encoder/second_layer/fully_connected/kernelFEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
д
0Encoder/second_layer/fully_connected/kernel/readIdentity+Encoder/second_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:

Ш
;Encoder/second_layer/fully_connected/bias/Initializer/zerosConst*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
е
)Encoder/second_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container *
shape:
Џ
0Encoder/second_layer/fully_connected/bias/AssignAssign)Encoder/second_layer/fully_connected/bias;Encoder/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Щ
.Encoder/second_layer/fully_connected/bias/readIdentity)Encoder/second_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
_output_shapes	
:
р
+Encoder/second_layer/fully_connected/MatMulMatMulEncoder/first_layer/leaky_relu0Encoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
о
,Encoder/second_layer/fully_connected/BiasAddBiasAdd+Encoder/second_layer/fully_connected/MatMul.Encoder/second_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
б
?Encoder/second_layer/batch_normalization/gamma/Initializer/onesConst*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
п
.Encoder/second_layer/batch_normalization/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container 
Т
5Encoder/second_layer/batch_normalization/gamma/AssignAssign.Encoder/second_layer/batch_normalization/gamma?Encoder/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
и
3Encoder/second_layer/batch_normalization/gamma/readIdentity.Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma
а
?Encoder/second_layer/batch_normalization/beta/Initializer/zerosConst*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
н
-Encoder/second_layer/batch_normalization/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
	container 
П
4Encoder/second_layer/batch_normalization/beta/AssignAssign-Encoder/second_layer/batch_normalization/beta?Encoder/second_layer/batch_normalization/beta/Initializer/zeros*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
е
2Encoder/second_layer/batch_normalization/beta/readIdentity-Encoder/second_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:
о
FEncoder/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ы
4Encoder/second_layer/batch_normalization/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
	container *
shape:
л
;Encoder/second_layer/batch_normalization/moving_mean/AssignAssign4Encoder/second_layer/batch_normalization/moving_meanFEncoder/second_layer/batch_normalization/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean
ъ
9Encoder/second_layer/batch_normalization/moving_mean/readIdentity4Encoder/second_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
_output_shapes	
:
х
IEncoder/second_layer/batch_normalization/moving_variance/Initializer/onesConst*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ѓ
8Encoder/second_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
	container *
shape:
ъ
?Encoder/second_layer/batch_normalization/moving_variance/AssignAssign8Encoder/second_layer/batch_normalization/moving_varianceIEncoder/second_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:
і
=Encoder/second_layer/batch_normalization/moving_variance/readIdentity8Encoder/second_layer/batch_normalization/moving_variance*
_output_shapes	
:*
T0*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance
}
8Encoder/second_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
м
6Encoder/second_layer/batch_normalization/batchnorm/addAdd=Encoder/second_layer/batch_normalization/moving_variance/read8Encoder/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:

8Encoder/second_layer/batch_normalization/batchnorm/RsqrtRsqrt6Encoder/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
в
6Encoder/second_layer/batch_normalization/batchnorm/mulMul8Encoder/second_layer/batch_normalization/batchnorm/Rsqrt3Encoder/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
и
8Encoder/second_layer/batch_normalization/batchnorm/mul_1Mul,Encoder/second_layer/fully_connected/BiasAdd6Encoder/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
и
8Encoder/second_layer/batch_normalization/batchnorm/mul_2Mul9Encoder/second_layer/batch_normalization/moving_mean/read6Encoder/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
б
6Encoder/second_layer/batch_normalization/batchnorm/subSub2Encoder/second_layer/batch_normalization/beta/read8Encoder/second_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ф
8Encoder/second_layer/batch_normalization/batchnorm/add_1Add8Encoder/second_layer/batch_normalization/batchnorm/mul_16Encoder/second_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:џџџџџџџџџ
j
%Encoder/second_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
О
#Encoder/second_layer/leaky_relu/mulMul%Encoder/second_layer/leaky_relu/alpha8Encoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
М
Encoder/second_layer/leaky_reluMaximum#Encoder/second_layer/leaky_relu/mul8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
Й
:Encoder/encoder_mu/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
Ћ
8Encoder/encoder_mu/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *?ШЪН*
dtype0*
_output_shapes
: 
Ћ
8Encoder/encoder_mu/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *?ШЪ=*
dtype0*
_output_shapes
: 

BEncoder/encoder_mu/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Encoder/encoder_mu/kernel/Initializer/random_uniform/shape*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
seed2 *
dtype0*
_output_shapes
:	d*

seed 

8Encoder/encoder_mu/kernel/Initializer/random_uniform/subSub8Encoder/encoder_mu/kernel/Initializer/random_uniform/max8Encoder/encoder_mu/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
: 

8Encoder/encoder_mu/kernel/Initializer/random_uniform/mulMulBEncoder/encoder_mu/kernel/Initializer/random_uniform/RandomUniform8Encoder/encoder_mu/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	d

4Encoder/encoder_mu/kernel/Initializer/random_uniformAdd8Encoder/encoder_mu/kernel/Initializer/random_uniform/mul8Encoder/encoder_mu/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	d
Н
Encoder/encoder_mu/kernel
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container *
shape:	d
ќ
 Encoder/encoder_mu/kernel/AssignAssignEncoder/encoder_mu/kernel4Encoder/encoder_mu/kernel/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	d

Encoder/encoder_mu/kernel/readIdentityEncoder/encoder_mu/kernel*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	d
Ђ
)Encoder/encoder_mu/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:d**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    
Џ
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
ц
Encoder/encoder_mu/bias/AssignAssignEncoder/encoder_mu/bias)Encoder/encoder_mu/bias/Initializer/zeros*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
validate_shape(*
_output_shapes
:d*
use_locking(

Encoder/encoder_mu/bias/readIdentityEncoder/encoder_mu/bias*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d
М
Encoder/encoder_mu/MatMulMatMulEncoder/second_layer/leaky_reluEncoder/encoder_mu/kernel/read*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b( *
T0
Ї
Encoder/encoder_mu/BiasAddBiasAddEncoder/encoder_mu/MatMulEncoder/encoder_mu/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџd
С
>Encoder/encoder_logvar/kernel/Initializer/random_uniform/shapeConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
Г
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *?ШЪН
Г
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *?ШЪ=

FEncoder/encoder_logvar/kernel/Initializer/random_uniform/RandomUniformRandomUniform>Encoder/encoder_logvar/kernel/Initializer/random_uniform/shape*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
seed2 *
dtype0*
_output_shapes
:	d*

seed 

<Encoder/encoder_logvar/kernel/Initializer/random_uniform/subSub<Encoder/encoder_logvar/kernel/Initializer/random_uniform/max<Encoder/encoder_logvar/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
: 
Ѕ
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/mulMulFEncoder/encoder_logvar/kernel/Initializer/random_uniform/RandomUniform<Encoder/encoder_logvar/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	d

8Encoder/encoder_logvar/kernel/Initializer/random_uniformAdd<Encoder/encoder_logvar/kernel/Initializer/random_uniform/mul<Encoder/encoder_logvar/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	d
Х
Encoder/encoder_logvar/kernel
VariableV2*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d

$Encoder/encoder_logvar/kernel/AssignAssignEncoder/encoder_logvar/kernel8Encoder/encoder_logvar/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel
Љ
"Encoder/encoder_logvar/kernel/readIdentityEncoder/encoder_logvar/kernel*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	d
Њ
-Encoder/encoder_logvar/bias/Initializer/zerosConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0*
_output_shapes
:d
З
Encoder/encoder_logvar/bias
VariableV2*
shared_name *.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d
і
"Encoder/encoder_logvar/bias/AssignAssignEncoder/encoder_logvar/bias-Encoder/encoder_logvar/bias/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
:d

 Encoder/encoder_logvar/bias/readIdentityEncoder/encoder_logvar/bias*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
:d
Ф
Encoder/encoder_logvar/MatMulMatMulEncoder/second_layer/leaky_relu"Encoder/encoder_logvar/kernel/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_a( 
Г
Encoder/encoder_logvar/BiasAddBiasAddEncoder/encoder_logvar/MatMul Encoder/encoder_logvar/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџd
e
Encoder/random_normal/shapeConst*
valueB	Rd*
dtype0	*
_output_shapes
:
_
Encoder/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
a
Encoder/random_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Њ
*Encoder/random_normal/RandomStandardNormalRandomStandardNormalEncoder/random_normal/shape*
dtype0*
_output_shapes
:d*
seed2 *

seed *
T0	

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
Encoder/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 

Encoder/truedivRealDivEncoder/encoder_logvar/BiasAddEncoder/truediv/y*'
_output_shapes
:џџџџџџџџџd*
T0
U
Encoder/ExpExpEncoder/truediv*'
_output_shapes
:џџџџџџџџџd*
T0
o
Encoder/logvar_stdMulEncoder/random_normalEncoder/Exp*
T0*'
_output_shapes
:џџџџџџџџџd
t
Encoder/AddAddEncoder/logvar_stdEncoder/encoder_mu/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџd
^
Encoder/encoder_codeSigmoidEncoder/Add*
T0*'
_output_shapes
:џџџџџџџџџd
л
KDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
Э
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *?ШЪН*
dtype0*
_output_shapes
: 
Э
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *?ШЪ=*
dtype0*
_output_shapes
: 
Р
SDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformKDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d*

seed *
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
seed2 
Ц
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
: 
й
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulSDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d
Ы
EDecoder/first_layer/fully_connected/kernel/Initializer/random_uniformAddIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
:	d*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel
п
*Decoder/first_layer/fully_connected/kernel
VariableV2*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name 
Р
1Decoder/first_layer/fully_connected/kernel/AssignAssign*Decoder/first_layer/fully_connected/kernelEDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d
а
/Decoder/first_layer/fully_connected/kernel/readIdentity*Decoder/first_layer/fully_connected/kernel*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d
Ц
:Decoder/first_layer/fully_connected/bias/Initializer/zerosConst*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
г
(Decoder/first_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
	container *
shape:
Ћ
/Decoder/first_layer/fully_connected/bias/AssignAssign(Decoder/first_layer/fully_connected/bias:Decoder/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ц
-Decoder/first_layer/fully_connected/bias/readIdentity(Decoder/first_layer/fully_connected/bias*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
_output_shapes	
:
д
*Decoder/first_layer/fully_connected/MatMulMatMulEncoder/encoder_code/Decoder/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
л
+Decoder/first_layer/fully_connected/BiasAddBiasAdd*Decoder/first_layer/fully_connected/MatMul-Decoder/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
i
$Decoder/first_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Џ
"Decoder/first_layer/leaky_relu/mulMul$Decoder/first_layer/leaky_relu/alpha+Decoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
­
Decoder/first_layer/leaky_reluMaximum"Decoder/first_layer/leaky_relu/mul+Decoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
н
LDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Я
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *qФН*
dtype0*
_output_shapes
: 
Я
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *qФ=*
dtype0*
_output_shapes
: 
Ф
TDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ъ
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
_output_shapes
: 
о
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel
а
FDecoder/second_layer/fully_connected/kernel/Initializer/random_uniformAddJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:

у
+Decoder/second_layer/fully_connected/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel
Х
2Decoder/second_layer/fully_connected/kernel/AssignAssign+Decoder/second_layer/fully_connected/kernelFDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
д
0Decoder/second_layer/fully_connected/kernel/readIdentity+Decoder/second_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:

Ш
;Decoder/second_layer/fully_connected/bias/Initializer/zerosConst*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
е
)Decoder/second_layer/fully_connected/bias
VariableV2*
shared_name *<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Џ
0Decoder/second_layer/fully_connected/bias/AssignAssign)Decoder/second_layer/fully_connected/bias;Decoder/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Щ
.Decoder/second_layer/fully_connected/bias/readIdentity)Decoder/second_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
_output_shapes	
:
р
+Decoder/second_layer/fully_connected/MatMulMatMulDecoder/first_layer/leaky_relu0Decoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
о
,Decoder/second_layer/fully_connected/BiasAddBiasAdd+Decoder/second_layer/fully_connected/MatMul.Decoder/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
б
?Decoder/second_layer/batch_normalization/gamma/Initializer/onesConst*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
п
.Decoder/second_layer/batch_normalization/gamma
VariableV2*
shared_name *A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
Т
5Decoder/second_layer/batch_normalization/gamma/AssignAssign.Decoder/second_layer/batch_normalization/gamma?Decoder/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
и
3Decoder/second_layer/batch_normalization/gamma/readIdentity.Decoder/second_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
_output_shapes	
:
а
?Decoder/second_layer/batch_normalization/beta/Initializer/zerosConst*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
н
-Decoder/second_layer/batch_normalization/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
	container 
П
4Decoder/second_layer/batch_normalization/beta/AssignAssign-Decoder/second_layer/batch_normalization/beta?Decoder/second_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
е
2Decoder/second_layer/batch_normalization/beta/readIdentity-Decoder/second_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
_output_shapes	
:
о
FDecoder/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ы
4Decoder/second_layer/batch_normalization/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
	container *
shape:
л
;Decoder/second_layer/batch_normalization/moving_mean/AssignAssign4Decoder/second_layer/batch_normalization/moving_meanFDecoder/second_layer/batch_normalization/moving_mean/Initializer/zeros*
T0*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ъ
9Decoder/second_layer/batch_normalization/moving_mean/readIdentity4Decoder/second_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
_output_shapes	
:
х
IDecoder/second_layer/batch_normalization/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
valueB*  ?
ѓ
8Decoder/second_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
	container *
shape:
ъ
?Decoder/second_layer/batch_normalization/moving_variance/AssignAssign8Decoder/second_layer/batch_normalization/moving_varianceIDecoder/second_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:
і
=Decoder/second_layer/batch_normalization/moving_variance/readIdentity8Decoder/second_layer/batch_normalization/moving_variance*
_output_shapes	
:*
T0*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance
}
8Decoder/second_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
м
6Decoder/second_layer/batch_normalization/batchnorm/addAdd=Decoder/second_layer/batch_normalization/moving_variance/read8Decoder/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:

8Decoder/second_layer/batch_normalization/batchnorm/RsqrtRsqrt6Decoder/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
в
6Decoder/second_layer/batch_normalization/batchnorm/mulMul8Decoder/second_layer/batch_normalization/batchnorm/Rsqrt3Decoder/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
и
8Decoder/second_layer/batch_normalization/batchnorm/mul_1Mul,Decoder/second_layer/fully_connected/BiasAdd6Decoder/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:џџџџџџџџџ*
T0
и
8Decoder/second_layer/batch_normalization/batchnorm/mul_2Mul9Decoder/second_layer/batch_normalization/moving_mean/read6Decoder/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
б
6Decoder/second_layer/batch_normalization/batchnorm/subSub2Decoder/second_layer/batch_normalization/beta/read8Decoder/second_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ф
8Decoder/second_layer/batch_normalization/batchnorm/add_1Add8Decoder/second_layer/batch_normalization/batchnorm/mul_16Decoder/second_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:џџџџџџџџџ*
T0
j
%Decoder/second_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>
О
#Decoder/second_layer/leaky_relu/mulMul%Decoder/second_layer/leaky_relu/alpha8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
М
Decoder/second_layer/leaky_reluMaximum#Decoder/second_layer/leaky_relu/mul8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
Џ
5Decoder/dense/kernel/Initializer/random_uniform/shapeConst*'
_class
loc:@Decoder/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ё
3Decoder/dense/kernel/Initializer/random_uniform/minConst*'
_class
loc:@Decoder/dense/kernel*
valueB
 *HYН*
dtype0*
_output_shapes
: 
Ё
3Decoder/dense/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@Decoder/dense/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
џ
=Decoder/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5Decoder/dense/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@Decoder/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ю
3Decoder/dense/kernel/Initializer/random_uniform/subSub3Decoder/dense/kernel/Initializer/random_uniform/max3Decoder/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@Decoder/dense/kernel*
_output_shapes
: 

3Decoder/dense/kernel/Initializer/random_uniform/mulMul=Decoder/dense/kernel/Initializer/random_uniform/RandomUniform3Decoder/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:

є
/Decoder/dense/kernel/Initializer/random_uniformAdd3Decoder/dense/kernel/Initializer/random_uniform/mul3Decoder/dense/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*'
_class
loc:@Decoder/dense/kernel
Е
Decoder/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *'
_class
loc:@Decoder/dense/kernel*
	container *
shape:

щ
Decoder/dense/kernel/AssignAssignDecoder/dense/kernel/Decoder/dense/kernel/Initializer/random_uniform*
T0*'
_class
loc:@Decoder/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

Decoder/dense/kernel/readIdentityDecoder/dense/kernel* 
_output_shapes
:
*
T0*'
_class
loc:@Decoder/dense/kernel

$Decoder/dense/bias/Initializer/zerosConst*%
_class
loc:@Decoder/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ї
Decoder/dense/bias
VariableV2*
shared_name *%
_class
loc:@Decoder/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
г
Decoder/dense/bias/AssignAssignDecoder/dense/bias$Decoder/dense/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(*
_output_shapes	
:

Decoder/dense/bias/readIdentityDecoder/dense/bias*
T0*%
_class
loc:@Decoder/dense/bias*
_output_shapes	
:
Г
Decoder/dense/MatMulMatMulDecoder/second_layer/leaky_reluDecoder/dense/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 

Decoder/dense/BiasAddBiasAddDecoder/dense/MatMulDecoder/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
d
Decoder/last_layerTanhDecoder/dense/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
t
Decoder/reshape_image/shapeConst*
dtype0*
_output_shapes
:*%
valueB"џџџџ         

Decoder/reshape_imageReshapeDecoder/last_layerDecoder/reshape_image/shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
~
Discriminator/noise_code_inPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџd*
shape:џџџџџџџџџd
ч
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
й
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *?ШЪН*
dtype0*
_output_shapes
: 
й
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *?ШЪ=*
dtype0*
_output_shapes
: 
в
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d*

seed *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
seed2 
о
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
ё
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d
у
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
:	d*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
ы
0Discriminator/first_layer/fully_connected/kernel
VariableV2*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d
и
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d
т
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d
в
@Discriminator/first_layer/fully_connected/bias/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
п
.Discriminator/first_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:
У
5Discriminator/first_layer/fully_connected/bias/AssignAssign.Discriminator/first_layer/fully_connected/bias@Discriminator/first_layer/fully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
и
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:
р
0Discriminator/first_layer/fully_connected/MatMulMatMulEncoder/encoder_code5Discriminator/first_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
э
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
o
*Discriminator/first_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
С
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
П
$Discriminator/first_layer/leaky_reluMaximum(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
щ
RDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
л
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *ѓЕН*
dtype0*
_output_shapes
: 
л
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *ѓЕ=*
dtype0*
_output_shapes
: 
ж
ZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformRDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
seed2 
т
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
_output_shapes
: 
і
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

ш
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

я
1Discriminator/second_layer/fully_connected/kernel
VariableV2*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
н
8Discriminator/second_layer/fully_connected/kernel/AssignAssign1Discriminator/second_layer/fully_connected/kernelLDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ц
6Discriminator/second_layer/fully_connected/kernel/readIdentity1Discriminator/second_layer/fully_connected/kernel*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

д
ADiscriminator/second_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    
с
/Discriminator/second_layer/fully_connected/bias
VariableV2*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ч
6Discriminator/second_layer/fully_connected/bias/AssignAssign/Discriminator/second_layer/fully_connected/biasADiscriminator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
л
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
ђ
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
№
2Discriminator/second_layer/fully_connected/BiasAddBiasAdd1Discriminator/second_layer/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
p
+Discriminator/second_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>
Ф
)Discriminator/second_layer/leaky_relu/mulMul+Discriminator/second_layer/leaky_relu/alpha2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Т
%Discriminator/second_layer/leaky_reluMaximum)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Й
:Discriminator/prob/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@Discriminator/prob/kernel*
valueB"      
Ћ
8Discriminator/prob/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *IvО*
dtype0*
_output_shapes
: 
Ћ
8Discriminator/prob/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 

BDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Discriminator/prob/kernel/Initializer/random_uniform/shape*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
seed2 *
dtype0*
_output_shapes
:	*

seed 

8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
: 

8Discriminator/prob/kernel/Initializer/random_uniform/mulMulBDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniform8Discriminator/prob/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*,
_class"
 loc:@Discriminator/prob/kernel

4Discriminator/prob/kernel/Initializer/random_uniformAdd8Discriminator/prob/kernel/Initializer/random_uniform/mul8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	
Н
Discriminator/prob/kernel
VariableV2*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel
ќ
 Discriminator/prob/kernel/AssignAssignDiscriminator/prob/kernel4Discriminator/prob/kernel/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	

Discriminator/prob/kernel/readIdentityDiscriminator/prob/kernel*
_output_shapes
:	*
T0*,
_class"
 loc:@Discriminator/prob/kernel
Ђ
)Discriminator/prob/bias/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
Џ
Discriminator/prob/bias
VariableV2**
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
ц
Discriminator/prob/bias/AssignAssignDiscriminator/prob/bias)Discriminator/prob/bias/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:

Discriminator/prob/bias/readIdentityDiscriminator/prob/bias*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
Т
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
Ї
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
щ
2Discriminator/first_layer_1/fully_connected/MatMulMatMulDiscriminator/noise_code_in5Discriminator/first_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
ё
3Discriminator/first_layer_1/fully_connected/BiasAddBiasAdd2Discriminator/first_layer_1/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
q
,Discriminator/first_layer_1/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ч
*Discriminator/first_layer_1/leaky_relu/mulMul,Discriminator/first_layer_1/leaky_relu/alpha3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
Х
&Discriminator/first_layer_1/leaky_reluMaximum*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
і
3Discriminator/second_layer_1/fully_connected/MatMulMatMul&Discriminator/first_layer_1/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
є
4Discriminator/second_layer_1/fully_connected/BiasAddBiasAdd3Discriminator/second_layer_1/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
r
-Discriminator/second_layer_1/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ъ
+Discriminator/second_layer_1/leaky_relu/mulMul-Discriminator/second_layer_1/leaky_relu/alpha4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
Ш
'Discriminator/second_layer_1/leaky_reluMaximum+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Ц
Discriminator/prob_1/MatMulMatMul'Discriminator/second_layer_1/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ћ
Discriminator/prob_1/BiasAddBiasAddDiscriminator/prob_1/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
i
ones_like/ShapeShapeDiscriminator/prob/BiasAdd*
_output_shapes
:*
T0*
out_type0
T
ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
w
	ones_likeFillones_like/Shapeones_like/Const*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
s
logistic_loss/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0

logistic_loss/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Ђ
logistic_loss/SelectSelectlogistic_loss/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
f
logistic_loss/NegNegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/NegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
q
logistic_loss/mulMulDiscriminator/prob/BiasAdd	ones_like*
T0*'
_output_shapes
:џџџџџџџџџ
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0*'
_output_shapes
:џџџџџџџџџ
b
logistic_loss/ExpExplogistic_loss/Select_1*
T0*'
_output_shapes
:џџџџџџџџџ
a
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*'
_output_shapes
:џџџџџџџџџ
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0*'
_output_shapes
:џџџџџџџџџ
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
m
adversalrial_lossMeanlogistic_lossConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
l
SubSubDecoder/reshape_imageEncoder/real_in*
T0*/
_output_shapes
:џџџџџџџџџ
I
AbsAbsSub*
T0*/
_output_shapes
:џџџџџџџџџ
`
Const_1Const*%
valueB"             *
dtype0*
_output_shapes
:
b
pixelwise_lossMeanAbsConst_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
J
mul/xConst*
valueB
 *o:*
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
 *wО?*
dtype0*
_output_shapes
: 
F
mul_1Mulmul_1/xpixelwise_loss*
_output_shapes
: *
T0
B
generator_lossAddmulmul_1*
_output_shapes
: *
T0
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
ones_like_1/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
ones_like_1Fillones_like_1/Shapeones_like_1/Const*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
w
logistic_loss_1/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0

logistic_loss_1/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
j
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Ѕ
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
w
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAddones_like_1*'
_output_shapes
:џџџџџџџџџ*
T0
y
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*'
_output_shapes
:џџџџџџџџџ*
T0
f
logistic_loss_1/ExpExplogistic_loss_1/Select_1*
T0*'
_output_shapes
:џџџџџџџџџ
e
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*'
_output_shapes
:џџџџџџџџџ*
T0
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*
T0*'
_output_shapes
:џџџџџџџџџ
X
Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
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
:џџџџџџџџџ
u
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0

logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Ј
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
h
logistic_loss_2/NegNegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
t
logistic_loss_2/mulMulDiscriminator/prob/BiasAdd
zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
y
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*'
_output_shapes
:џџџџџџџџџ*
T0
f
logistic_loss_2/ExpExplogistic_loss_2/Select_1*
T0*'
_output_shapes
:џџџџџџџџџ
e
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*
T0*'
_output_shapes
:џџџџџџџџџ
t
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*
T0*'
_output_shapes
:џџџџџџџџџ
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
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
<
#gradients/add_grad/tuple/group_depsNoOp^gradients/Fill
Б
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Fill$^gradients/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
Г
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
­
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

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

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

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

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

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
t
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Г
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
Ђ
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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

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
 
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
a
gradients/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 

gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
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
и
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
М
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Л
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Р
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
С
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

/gradients/logistic_loss_1_grad/tuple/group_depsNoOp'^gradients/logistic_loss_1_grad/Reshape)^gradients/logistic_loss_1_grad/Reshape_1

7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape

9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1
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
и
4gradients/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_2_grad/Shape&gradients/logistic_loss_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
О
"gradients/logistic_loss_2_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
&gradients/logistic_loss_2_grad/ReshapeReshape"gradients/logistic_loss_2_grad/Sum$gradients/logistic_loss_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Т
$gradients/logistic_loss_2_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
С
(gradients/logistic_loss_2_grad/Reshape_1Reshape$gradients/logistic_loss_2_grad/Sum_1&gradients/logistic_loss_2_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

/gradients/logistic_loss_2_grad/tuple/group_depsNoOp'^gradients/logistic_loss_2_grad/Reshape)^gradients/logistic_loss_2_grad/Reshape_1

7gradients/logistic_loss_2_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_2_grad/Reshape0^gradients/logistic_loss_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*9
_class/
-+loc:@gradients/logistic_loss_2_grad/Reshape

9gradients/logistic_loss_2_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_2_grad/Reshape_10^gradients/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
~
(gradients/logistic_loss_1/sub_grad/ShapeShapelogistic_loss_1/Select*
_output_shapes
:*
T0*
out_type0
}
*gradients/logistic_loss_1/sub_grad/Shape_1Shapelogistic_loss_1/mul*
_output_shapes
:*
T0*
out_type0
ф
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
р
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ч
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ф
(gradients/logistic_loss_1/sub_grad/Sum_1Sum7gradients/logistic_loss_1_grad/tuple/control_dependency:gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
z
&gradients/logistic_loss_1/sub_grad/NegNeg(gradients/logistic_loss_1/sub_grad/Sum_1*
_output_shapes
:*
T0
Ы
,gradients/logistic_loss_1/sub_grad/Reshape_1Reshape&gradients/logistic_loss_1/sub_grad/Neg*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3gradients/logistic_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/sub_grad/Reshape-^gradients/logistic_loss_1/sub_grad/Reshape_1

;gradients/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/sub_grad/Reshape4^gradients/logistic_loss_1/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/sub_grad/Reshape
 
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ћ
*gradients/logistic_loss_1/Log1p_grad/add/xConst:^gradients/logistic_loss_1_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ђ
(gradients/logistic_loss_1/Log1p_grad/addAdd*gradients/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*'
_output_shapes
:џџџџџџџџџ*
T0
Э
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ
~
(gradients/logistic_loss_2/sub_grad/ShapeShapelogistic_loss_2/Select*
_output_shapes
:*
T0*
out_type0
}
*gradients/logistic_loss_2/sub_grad/Shape_1Shapelogistic_loss_2/mul*
T0*
out_type0*
_output_shapes
:
ф
8gradients/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_2/sub_grad/Shape*gradients/logistic_loss_2/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
р
&gradients/logistic_loss_2/sub_grad/SumSum7gradients/logistic_loss_2_grad/tuple/control_dependency8gradients/logistic_loss_2/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ч
*gradients/logistic_loss_2/sub_grad/ReshapeReshape&gradients/logistic_loss_2/sub_grad/Sum(gradients/logistic_loss_2/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ф
(gradients/logistic_loss_2/sub_grad/Sum_1Sum7gradients/logistic_loss_2_grad/tuple/control_dependency:gradients/logistic_loss_2/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
z
&gradients/logistic_loss_2/sub_grad/NegNeg(gradients/logistic_loss_2/sub_grad/Sum_1*
T0*
_output_shapes
:
Ы
,gradients/logistic_loss_2/sub_grad/Reshape_1Reshape&gradients/logistic_loss_2/sub_grad/Neg*gradients/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3gradients/logistic_loss_2/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_2/sub_grad/Reshape-^gradients/logistic_loss_2/sub_grad/Reshape_1

;gradients/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_2/sub_grad/Reshape4^gradients/logistic_loss_2/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*=
_class3
1/loc:@gradients/logistic_loss_2/sub_grad/Reshape
 
=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_2/sub_grad/Reshape_14^gradients/logistic_loss_2/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/logistic_loss_2/sub_grad/Reshape_1
Ћ
*gradients/logistic_loss_2/Log1p_grad/add/xConst:^gradients/logistic_loss_2_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ђ
(gradients/logistic_loss_2/Log1p_grad/addAdd*gradients/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_2/Log1p_grad/add*'
_output_shapes
:џџџџџџџџџ*
T0
Э
(gradients/logistic_loss_2/Log1p_grad/mulMul9gradients/logistic_loss_2_grad/tuple/control_dependency_1/gradients/logistic_loss_2/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ

0gradients/logistic_loss_1/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
ѕ
,gradients/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual;gradients/logistic_loss_1/sub_grad/tuple/control_dependency0gradients/logistic_loss_1/Select_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
ї
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0

6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
Є
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select
Њ
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_grad/Select_1

(gradients/logistic_loss_1/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
_output_shapes
:*
T0*
out_type0
u
*gradients/logistic_loss_1/mul_grad/Shape_1Shapeones_like_1*
T0*
out_type0*
_output_shapes
:
ф
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1ones_like_1*'
_output_shapes
:џџџџџџџџџ*
T0
Я
&gradients/logistic_loss_1/mul_grad/SumSum&gradients/logistic_loss_1/mul_grad/Mul8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ч
*gradients/logistic_loss_1/mul_grad/ReshapeReshape&gradients/logistic_loss_1/mul_grad/Sum(gradients/logistic_loss_1/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
О
(gradients/logistic_loss_1/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
е
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Э
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3gradients/logistic_loss_1/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/mul_grad/Reshape-^gradients/logistic_loss_1/mul_grad/Reshape_1

;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/mul_grad/Reshape4^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
 
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

0gradients/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
ѕ
,gradients/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual;gradients/logistic_loss_2/sub_grad/tuple/control_dependency0gradients/logistic_loss_2/Select_grad/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
ї
.gradients/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual0gradients/logistic_loss_2/Select_grad/zeros_like;gradients/logistic_loss_2/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

6gradients/logistic_loss_2/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_2/Select_grad/Select/^gradients/logistic_loss_2/Select_grad/Select_1
Є
>gradients/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_2/Select_grad/Select7^gradients/logistic_loss_2/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Њ
@gradients/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_2/Select_grad/Select_17^gradients/logistic_loss_2/Select_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@gradients/logistic_loss_2/Select_grad/Select_1

(gradients/logistic_loss_2/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
t
*gradients/logistic_loss_2/mul_grad/Shape_1Shape
zeros_like*
_output_shapes
:*
T0*
out_type0
ф
8gradients/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_2/mul_grad/Shape*gradients/logistic_loss_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Њ
&gradients/logistic_loss_2/mul_grad/MulMul=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1
zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Я
&gradients/logistic_loss_2/mul_grad/SumSum&gradients/logistic_loss_2/mul_grad/Mul8gradients/logistic_loss_2/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ч
*gradients/logistic_loss_2/mul_grad/ReshapeReshape&gradients/logistic_loss_2/mul_grad/Sum(gradients/logistic_loss_2/mul_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
М
(gradients/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
е
(gradients/logistic_loss_2/mul_grad/Sum_1Sum(gradients/logistic_loss_2/mul_grad/Mul_1:gradients/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Э
,gradients/logistic_loss_2/mul_grad/Reshape_1Reshape(gradients/logistic_loss_2/mul_grad/Sum_1*gradients/logistic_loss_2/mul_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

3gradients/logistic_loss_2/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_2/mul_grad/Reshape-^gradients/logistic_loss_2/mul_grad/Reshape_1

;gradients/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_2/mul_grad/Reshape4^gradients/logistic_loss_2/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_2/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
 
=gradients/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_2/mul_grad/Reshape_14^gradients/logistic_loss_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

&gradients/logistic_loss_2/Exp_grad/mulMul(gradients/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*'
_output_shapes
:џџџџџџџџџ*
T0
ф
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
ц
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Є
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
Ќ
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select*'
_output_shapes
:џџџџџџџџџ
В
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1*'
_output_shapes
:џџџџџџџџџ

2gradients/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*
T0*'
_output_shapes
:џџџџџџџџџ
ф
.gradients/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual&gradients/logistic_loss_2/Exp_grad/mul2gradients/logistic_loss_2/Select_1_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
ц
0gradients/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients/logistic_loss_2/Select_1_grad/zeros_like&gradients/logistic_loss_2/Exp_grad/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Є
8gradients/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_2/Select_1_grad/Select1^gradients/logistic_loss_2/Select_1_grad/Select_1
Ќ
@gradients/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_2/Select_1_grad/Select9^gradients/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_2/Select_1_grad/Select*'
_output_shapes
:џџџџџџџџџ
В
Bgradients/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_2/Select_1_grad/Select_19^gradients/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_2/Select_1_grad/Select_1*'
_output_shapes
:џџџџџџџџџ
Ё
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
&gradients/logistic_loss_2/Neg_grad/NegNeg@gradients/logistic_loss_2/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ћ
gradients/AddNAddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N*'
_output_shapes
:џџџџџџџџџ

7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:

<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN8^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad

Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Л
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
§
gradients/AddN_1AddN>gradients/logistic_loss_2/Select_grad/tuple/control_dependency;gradients/logistic_loss_2/mul_grad/tuple/control_dependencyBgradients/logistic_loss_2/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_2/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*
N*'
_output_shapes
:џџџџџџџџџ

5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
data_formatNHWC*
_output_shapes
:*
T0

:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_16^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad

Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Г
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
њ
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ќ
3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
­
;gradients/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp2^gradients/Discriminator/prob_1/MatMul_grad/MatMul4^gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
Й
Cgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/Discriminator/prob_1/MatMul_grad/MatMul<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
Ж
Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
і
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
і
1gradients/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(
Ї
9gradients/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp0^gradients/Discriminator/prob/MatMul_grad/MatMul2^gradients/Discriminator/prob/MatMul_grad/MatMul_1
Б
Agradients/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity/gradients/Discriminator/prob/MatMul_grad/MatMul:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/Discriminator/prob/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
Ў
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	
 
gradients/AddN_2AddNFgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
Ї
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
В
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
С
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosFill>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
щ
Cgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
 
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Т
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
Ф
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape:gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
ы
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
Ѓ
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ў
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Н
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

:gradients/Discriminator/second_layer/leaky_relu_grad/zerosFill<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
у
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

Jgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Discriminator/second_layer/leaky_relu_grad/Shape<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
К
;gradients/Discriminator/second_layer/leaky_relu_grad/SelectSelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency:gradients/Discriminator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
М
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

8gradients/Discriminator/second_layer/leaky_relu_grad/SumSum;gradients/Discriminator/second_layer/leaky_relu_grad/SelectJgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ў
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1Lgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Э
Egradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp=^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape?^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
у
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
щ
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

gradients/AddN_3AddNEgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1*
N*
_output_shapes
:	*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1

@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ж
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ќ
Pgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
џ
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ў
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
њ
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Rgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Dgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1Reshape@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
п
Kgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeE^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
щ
Sgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeL^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 

Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1L^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
В
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
І
Ngradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulMulMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ј
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
є
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Pgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Bgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
й
Igradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpA^gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeC^gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
с
Qgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeJ^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
љ
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
г
gradients/AddN_4AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
­
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:
С
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
й
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Э
gradients/AddN_5AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Ћ
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:
Н
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
г
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1

\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Т
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ќ
Kgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ѕ
Sgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulL^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1

[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulT^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul

]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

О
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
І
Igradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_reluZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
я
Qgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpH^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulJ^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1

Ygradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityGgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulR^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Z
_classP
NLloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul

[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
щ
gradients/AddN_6AddN^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
Ѕ
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
А
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
и
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

;gradients/Discriminator/first_layer_1/leaky_relu_grad/zerosFill=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
ц
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
з
<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
й
>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients/Discriminator/first_layer_1/leaky_relu_grad/SumSum<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectKgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
а
Fgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp>^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape@^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
ч
Ngradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeG^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
э
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Ё
9gradients/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ќ
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
д
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2ShapeYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

9gradients/Discriminator/first_layer/leaky_relu_grad/zerosFill;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
р
@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
:gradients/Discriminator/first_layer/leaky_relu_grad/SelectSelect@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency9gradients/Discriminator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
б
<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Select@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqual9gradients/Discriminator/first_layer/leaky_relu_grad/zerosYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

7gradients/Discriminator/first_layer/leaky_relu_grad/SumSum:gradients/Discriminator/first_layer/leaky_relu_grad/SelectIgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ћ
;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape7gradients/Discriminator/first_layer/leaky_relu_grad/Sum9gradients/Discriminator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Kgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ъ
Dgradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp<^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape>^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
п
Lgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeE^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
х
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
ш
gradients/AddN_7AddN]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
N* 
_output_shapes
:
*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1

?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Д
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Љ
Ogradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ќ
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulOgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ћ
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ї
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Qgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Cgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1Reshape?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
м
Jgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeD^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
х
Rgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeK^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
§
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

=gradients/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
А
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ѓ
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
і
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ѕ
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ё
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Ogradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Agradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
ж
Hgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp@^gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeB^gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
н
Pgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeI^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
ѕ
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
а
gradients/AddN_8AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
Ќ
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:
П
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ж
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ъ
gradients/AddN_9AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Њ
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:
Л
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
а
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradR^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
О
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(*
T0

Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/noise_code_in[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d*
transpose_a(*
transpose_b( *
T0
ђ
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1

Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџd*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul

\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1S^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d
К
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(

Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_codeYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	d*
transpose_a(
ь
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1

Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџd

Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d
ч
gradients/AddN_10AddN]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
х
gradients/AddN_11AddN\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
N*
_output_shapes
:	d*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
Ё
beta1_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *   ?*
dtype0*
_output_shapes
: 
В
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
б
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 

beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
Ё
beta2_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *wО?*
dtype0*
_output_shapes
: 
В
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
б
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias

beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
ћ
eDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
х
[Discriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ђ
UDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zerosFilleDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensor[Discriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/Const*
_output_shapes
:	d*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0
ў
CDiscriminator/first_layer/fully_connected/kernel/discriminator_opti
VariableV2*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d

JDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/AssignAssignCDiscriminator/first_layer/fully_connected/kernel/discriminator_optiUDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel

HDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/readIdentityCDiscriminator/first_layer/fully_connected/kernel/discriminator_opti*
_output_shapes
:	d*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
§
gDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
ч
]Discriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ј
WDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zerosFillgDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensor]Discriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d

EDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1
VariableV2*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name 

LDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/AssignAssignEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1WDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel

JDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/readIdentityEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d
х
SDiscriminator/first_layer/fully_connected/bias/discriminator_opti/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ђ
ADiscriminator/first_layer/fully_connected/bias/discriminator_opti
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ќ
HDiscriminator/first_layer/fully_connected/bias/discriminator_opti/AssignAssignADiscriminator/first_layer/fully_connected/bias/discriminator_optiSDiscriminator/first_layer/fully_connected/bias/discriminator_opti/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ў
FDiscriminator/first_layer/fully_connected/bias/discriminator_opti/readIdentityADiscriminator/first_layer/fully_connected/bias/discriminator_opti*
_output_shapes	
:*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
ч
UDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
є
CDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container 

JDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/AssignAssignCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1UDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:

HDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/readIdentityCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:
§
fDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ч
\Discriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
VDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zerosFillfDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensor\Discriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:


DDiscriminator/second_layer/fully_connected/kernel/discriminator_opti
VariableV2*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:


KDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/AssignAssignDDiscriminator/second_layer/fully_connected/kernel/discriminator_optiVDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel

IDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/readIdentityDDiscriminator/second_layer/fully_connected/kernel/discriminator_opti*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

џ
hDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
щ
^Discriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
­
XDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zerosFillhDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensor^Discriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:


FDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1
VariableV2*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 

MDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/AssignAssignFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1XDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

KDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/readIdentityFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
ч
TDiscriminator/second_layer/fully_connected/bias/discriminator_opti/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
є
BDiscriminator/second_layer/fully_connected/bias/discriminator_opti
VariableV2*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:

IDiscriminator/second_layer/fully_connected/bias/discriminator_opti/AssignAssignBDiscriminator/second_layer/fully_connected/bias/discriminator_optiTDiscriminator/second_layer/fully_connected/bias/discriminator_opti/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:

GDiscriminator/second_layer/fully_connected/bias/discriminator_opti/readIdentityBDiscriminator/second_layer/fully_connected/bias/discriminator_opti*
_output_shapes	
:*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
щ
VDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
і
DDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1
VariableV2*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

KDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/AssignAssignDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1VDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

IDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/readIdentityDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1*
_output_shapes	
:*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
У
>Discriminator/prob/kernel/discriminator_opti/Initializer/zerosConst*
dtype0*
_output_shapes
:	*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    
а
,Discriminator/prob/kernel/discriminator_opti
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	
Ќ
3Discriminator/prob/kernel/discriminator_opti/AssignAssign,Discriminator/prob/kernel/discriminator_opti>Discriminator/prob/kernel/discriminator_opti/Initializer/zeros*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
У
1Discriminator/prob/kernel/discriminator_opti/readIdentity,Discriminator/prob/kernel/discriminator_opti*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	
Х
@Discriminator/prob/kernel/discriminator_opti_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
в
.Discriminator/prob/kernel/discriminator_opti_1
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container 
В
5Discriminator/prob/kernel/discriminator_opti_1/AssignAssign.Discriminator/prob/kernel/discriminator_opti_1@Discriminator/prob/kernel/discriminator_opti_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel
Ч
3Discriminator/prob/kernel/discriminator_opti_1/readIdentity.Discriminator/prob/kernel/discriminator_opti_1*
_output_shapes
:	*
T0*,
_class"
 loc:@Discriminator/prob/kernel
Е
<Discriminator/prob/bias/discriminator_opti/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
Т
*Discriminator/prob/bias/discriminator_opti
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias

1Discriminator/prob/bias/discriminator_opti/AssignAssign*Discriminator/prob/bias/discriminator_opti<Discriminator/prob/bias/discriminator_opti/Initializer/zeros*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:*
use_locking(
И
/Discriminator/prob/bias/discriminator_opti/readIdentity*Discriminator/prob/bias/discriminator_opti*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
З
>Discriminator/prob/bias/discriminator_opti_1/Initializer/zerosConst*
dtype0*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    
Ф
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
Ѕ
3Discriminator/prob/bias/discriminator_opti_1/AssignAssign,Discriminator/prob/bias/discriminator_opti_1>Discriminator/prob/bias/discriminator_opti_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
М
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
 *ЗQ9
]
discriminator_opti/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *   ?
]
discriminator_opti/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
_
discriminator_opti/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
о
Tdiscriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam0Discriminator/first_layer/fully_connected/kernelCDiscriminator/first_layer/fully_connected/kernel/discriminator_optiEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_11*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	d*
use_locking( 
а
Rdiscriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam.Discriminator/first_layer/fully_connected/biasADiscriminator/first_layer/fully_connected/bias/discriminator_optiCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_10*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:
у
Udiscriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernelDDiscriminator/second_layer/fully_connected/kernel/discriminator_optiFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_7*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:

д
Sdiscriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/biasBDiscriminator/second_layer/fully_connected/bias/discriminator_optiDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_6*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
ъ
=discriminator_opti/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernel,Discriminator/prob/kernel/discriminator_opti.Discriminator/prob/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_3*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
use_nesterov( *
_output_shapes
:	
л
;discriminator_opti/update_Discriminator/prob/bias/ApplyAdam	ApplyAdamDiscriminator/prob/bias*Discriminator/prob/bias/discriminator_opti,Discriminator/prob/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_2*
use_locking( *
T0**
_class 
loc:@Discriminator/prob/bias*
use_nesterov( *
_output_shapes
:

discriminator_opti/mulMulbeta1_power/readdiscriminator_opti/beta1S^discriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamU^discriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam<^discriminator_opti/update_Discriminator/prob/bias/ApplyAdam>^discriminator_opti/update_Discriminator/prob/kernel/ApplyAdamT^discriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamV^discriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
е
discriminator_opti/AssignAssignbeta1_powerdiscriminator_opti/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias

discriminator_opti/mul_1Mulbeta2_power/readdiscriminator_opti/beta2S^discriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamU^discriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam<^discriminator_opti/update_Discriminator/prob/bias/ApplyAdam>^discriminator_opti/update_Discriminator/prob/kernel/ApplyAdamT^discriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamV^discriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
й
discriminator_opti/Assign_1Assignbeta2_powerdiscriminator_opti/mul_1*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
Ќ
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
 *  ?*
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
Я
8gradients_1/generator_loss_grad/tuple/control_dependencyIdentitygradients_1/Fill1^gradients_1/generator_loss_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill*
_output_shapes
: 
б
:gradients_1/generator_loss_grad/tuple/control_dependency_1Identitygradients_1/Fill1^gradients_1/generator_loss_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill*
_output_shapes
: 

gradients_1/mul_grad/MulMul8gradients_1/generator_loss_grad/tuple/control_dependencyadversalrial_loss*
T0*
_output_shapes
: 

gradients_1/mul_grad/Mul_1Mul8gradients_1/generator_loss_grad/tuple/control_dependencymul/x*
_output_shapes
: *
T0
e
%gradients_1/mul_grad/tuple/group_depsNoOp^gradients_1/mul_grad/Mul^gradients_1/mul_grad/Mul_1
Щ
-gradients_1/mul_grad/tuple/control_dependencyIdentitygradients_1/mul_grad/Mul&^gradients_1/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients_1/mul_grad/Mul
Я
/gradients_1/mul_grad/tuple/control_dependency_1Identitygradients_1/mul_grad/Mul_1&^gradients_1/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_grad/Mul_1*
_output_shapes
: 

gradients_1/mul_1_grad/MulMul:gradients_1/generator_loss_grad/tuple/control_dependency_1pixelwise_loss*
T0*
_output_shapes
: 

gradients_1/mul_1_grad/Mul_1Mul:gradients_1/generator_loss_grad/tuple/control_dependency_1mul_1/x*
T0*
_output_shapes
: 
k
'gradients_1/mul_1_grad/tuple/group_depsNoOp^gradients_1/mul_1_grad/Mul^gradients_1/mul_1_grad/Mul_1
б
/gradients_1/mul_1_grad/tuple/control_dependencyIdentitygradients_1/mul_1_grad/Mul(^gradients_1/mul_1_grad/tuple/group_deps*
_output_shapes
: *
T0*-
_class#
!loc:@gradients_1/mul_1_grad/Mul
з
1gradients_1/mul_1_grad/tuple/control_dependency_1Identitygradients_1/mul_1_grad/Mul_1(^gradients_1/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_1_grad/Mul_1*
_output_shapes
: 

0gradients_1/adversalrial_loss_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Я
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
Щ
'gradients_1/adversalrial_loss_grad/TileTile*gradients_1/adversalrial_loss_grad/Reshape(gradients_1/adversalrial_loss_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
w
*gradients_1/adversalrial_loss_grad/Shape_1Shapelogistic_loss*
T0*
out_type0*
_output_shapes
:
m
*gradients_1/adversalrial_loss_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
r
(gradients_1/adversalrial_loss_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
У
'gradients_1/adversalrial_loss_grad/ProdProd*gradients_1/adversalrial_loss_grad/Shape_1(gradients_1/adversalrial_loss_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
t
*gradients_1/adversalrial_loss_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ч
)gradients_1/adversalrial_loss_grad/Prod_1Prod*gradients_1/adversalrial_loss_grad/Shape_2*gradients_1/adversalrial_loss_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
n
,gradients_1/adversalrial_loss_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Џ
*gradients_1/adversalrial_loss_grad/MaximumMaximum)gradients_1/adversalrial_loss_grad/Prod_1,gradients_1/adversalrial_loss_grad/Maximum/y*
T0*
_output_shapes
: 
­
+gradients_1/adversalrial_loss_grad/floordivFloorDiv'gradients_1/adversalrial_loss_grad/Prod*gradients_1/adversalrial_loss_grad/Maximum*
T0*
_output_shapes
: 

'gradients_1/adversalrial_loss_grad/CastCast+gradients_1/adversalrial_loss_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
Й
*gradients_1/adversalrial_loss_grad/truedivRealDiv'gradients_1/adversalrial_loss_grad/Tile'gradients_1/adversalrial_loss_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

-gradients_1/pixelwise_loss_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
г
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
Ш
$gradients_1/pixelwise_loss_grad/TileTile'gradients_1/pixelwise_loss_grad/Reshape%gradients_1/pixelwise_loss_grad/Shape*/
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
j
'gradients_1/pixelwise_loss_grad/Shape_1ShapeAbs*
T0*
out_type0*
_output_shapes
:
j
'gradients_1/pixelwise_loss_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
o
%gradients_1/pixelwise_loss_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
К
$gradients_1/pixelwise_loss_grad/ProdProd'gradients_1/pixelwise_loss_grad/Shape_1%gradients_1/pixelwise_loss_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
q
'gradients_1/pixelwise_loss_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
О
&gradients_1/pixelwise_loss_grad/Prod_1Prod'gradients_1/pixelwise_loss_grad/Shape_2'gradients_1/pixelwise_loss_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
k
)gradients_1/pixelwise_loss_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
І
'gradients_1/pixelwise_loss_grad/MaximumMaximum&gradients_1/pixelwise_loss_grad/Prod_1)gradients_1/pixelwise_loss_grad/Maximum/y*
T0*
_output_shapes
: 
Є
(gradients_1/pixelwise_loss_grad/floordivFloorDiv$gradients_1/pixelwise_loss_grad/Prod'gradients_1/pixelwise_loss_grad/Maximum*
T0*
_output_shapes
: 

$gradients_1/pixelwise_loss_grad/CastCast(gradients_1/pixelwise_loss_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
И
'gradients_1/pixelwise_loss_grad/truedivRealDiv$gradients_1/pixelwise_loss_grad/Tile$gradients_1/pixelwise_loss_grad/Cast*
T0*/
_output_shapes
:џџџџџџџџџ
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
и
4gradients_1/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_1/logistic_loss_grad/Shape&gradients_1/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ы
"gradients_1/logistic_loss_grad/SumSum*gradients_1/adversalrial_loss_grad/truediv4gradients_1/logistic_loss_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Л
&gradients_1/logistic_loss_grad/ReshapeReshape"gradients_1/logistic_loss_grad/Sum$gradients_1/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Я
$gradients_1/logistic_loss_grad/Sum_1Sum*gradients_1/adversalrial_loss_grad/truediv6gradients_1/logistic_loss_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(gradients_1/logistic_loss_grad/Reshape_1Reshape$gradients_1/logistic_loss_grad/Sum_1&gradients_1/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

/gradients_1/logistic_loss_grad/tuple/group_depsNoOp'^gradients_1/logistic_loss_grad/Reshape)^gradients_1/logistic_loss_grad/Reshape_1

7gradients_1/logistic_loss_grad/tuple/control_dependencyIdentity&gradients_1/logistic_loss_grad/Reshape0^gradients_1/logistic_loss_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*9
_class/
-+loc:@gradients_1/logistic_loss_grad/Reshape

9gradients_1/logistic_loss_grad/tuple/control_dependency_1Identity(gradients_1/logistic_loss_grad/Reshape_10^gradients_1/logistic_loss_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
`
gradients_1/Abs_grad/SignSignSub*
T0*/
_output_shapes
:џџџџџџџџџ

gradients_1/Abs_grad/mulMul'gradients_1/pixelwise_loss_grad/truedivgradients_1/Abs_grad/Sign*
T0*/
_output_shapes
:џџџџџџџџџ
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
ф
8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/sub_grad/Shape*gradients_1/logistic_loss/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
р
&gradients_1/logistic_loss/sub_grad/SumSum7gradients_1/logistic_loss_grad/tuple/control_dependency8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ч
*gradients_1/logistic_loss/sub_grad/ReshapeReshape&gradients_1/logistic_loss/sub_grad/Sum(gradients_1/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ф
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
Ы
,gradients_1/logistic_loss/sub_grad/Reshape_1Reshape&gradients_1/logistic_loss/sub_grad/Neg*gradients_1/logistic_loss/sub_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

3gradients_1/logistic_loss/sub_grad/tuple/group_depsNoOp+^gradients_1/logistic_loss/sub_grad/Reshape-^gradients_1/logistic_loss/sub_grad/Reshape_1

;gradients_1/logistic_loss/sub_grad/tuple/control_dependencyIdentity*gradients_1/logistic_loss/sub_grad/Reshape4^gradients_1/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
 
=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1Identity,gradients_1/logistic_loss/sub_grad/Reshape_14^gradients_1/logistic_loss/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ћ
*gradients_1/logistic_loss/Log1p_grad/add/xConst:^gradients_1/logistic_loss_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  ?
 
(gradients_1/logistic_loss/Log1p_grad/addAdd*gradients_1/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*'
_output_shapes
:џџџџџџџџџ*
T0

/gradients_1/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_1/logistic_loss/Log1p_grad/add*'
_output_shapes
:џџџџџџџџџ*
T0
Э
(gradients_1/logistic_loss/Log1p_grad/mulMul9gradients_1/logistic_loss_grad/tuple/control_dependency_1/gradients_1/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ
o
gradients_1/Sub_grad/ShapeShapeDecoder/reshape_image*
T0*
out_type0*
_output_shapes
:
k
gradients_1/Sub_grad/Shape_1ShapeEncoder/real_in*
T0*
out_type0*
_output_shapes
:
К
*gradients_1/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Sub_grad/Shapegradients_1/Sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
gradients_1/Sub_grad/SumSumgradients_1/Abs_grad/mul*gradients_1/Sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ѕ
gradients_1/Sub_grad/ReshapeReshapegradients_1/Sub_grad/Sumgradients_1/Sub_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
Љ
gradients_1/Sub_grad/Sum_1Sumgradients_1/Abs_grad/mul,gradients_1/Sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
^
gradients_1/Sub_grad/NegNeggradients_1/Sub_grad/Sum_1*
T0*
_output_shapes
:
Љ
gradients_1/Sub_grad/Reshape_1Reshapegradients_1/Sub_grad/Neggradients_1/Sub_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
m
%gradients_1/Sub_grad/tuple/group_depsNoOp^gradients_1/Sub_grad/Reshape^gradients_1/Sub_grad/Reshape_1
ъ
-gradients_1/Sub_grad/tuple/control_dependencyIdentitygradients_1/Sub_grad/Reshape&^gradients_1/Sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/Sub_grad/Reshape*/
_output_shapes
:џџџџџџџџџ
№
/gradients_1/Sub_grad/tuple/control_dependency_1Identitygradients_1/Sub_grad/Reshape_1&^gradients_1/Sub_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*1
_class'
%#loc:@gradients_1/Sub_grad/Reshape_1

0gradients_1/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
ѓ
,gradients_1/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual;gradients_1/logistic_loss/sub_grad/tuple/control_dependency0gradients_1/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
ѕ
.gradients_1/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_1/logistic_loss/Select_grad/zeros_like;gradients_1/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

6gradients_1/logistic_loss/Select_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss/Select_grad/Select/^gradients_1/logistic_loss/Select_grad/Select_1
Є
>gradients_1/logistic_loss/Select_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss/Select_grad/Select7^gradients_1/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Њ
@gradients_1/logistic_loss/Select_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss/Select_grad/Select_17^gradients_1/logistic_loss/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss/Select_grad/Select_1*'
_output_shapes
:џџџџџџџџџ

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
ф
8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/mul_grad/Shape*gradients_1/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Љ
&gradients_1/logistic_loss/mul_grad/MulMul=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*
T0*'
_output_shapes
:џџџџџџџџџ
Я
&gradients_1/logistic_loss/mul_grad/SumSum&gradients_1/logistic_loss/mul_grad/Mul8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ч
*gradients_1/logistic_loss/mul_grad/ReshapeReshape&gradients_1/logistic_loss/mul_grad/Sum(gradients_1/logistic_loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
М
(gradients_1/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
е
(gradients_1/logistic_loss/mul_grad/Sum_1Sum(gradients_1/logistic_loss/mul_grad/Mul_1:gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Э
,gradients_1/logistic_loss/mul_grad/Reshape_1Reshape(gradients_1/logistic_loss/mul_grad/Sum_1*gradients_1/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3gradients_1/logistic_loss/mul_grad/tuple/group_depsNoOp+^gradients_1/logistic_loss/mul_grad/Reshape-^gradients_1/logistic_loss/mul_grad/Reshape_1

;gradients_1/logistic_loss/mul_grad/tuple/control_dependencyIdentity*gradients_1/logistic_loss/mul_grad/Reshape4^gradients_1/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
 
=gradients_1/logistic_loss/mul_grad/tuple/control_dependency_1Identity,gradients_1/logistic_loss/mul_grad/Reshape_14^gradients_1/logistic_loss/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

&gradients_1/logistic_loss/Exp_grad/mulMul(gradients_1/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0*'
_output_shapes
:џџџџџџџџџ
~
,gradients_1/Decoder/reshape_image_grad/ShapeShapeDecoder/last_layer*
T0*
out_type0*
_output_shapes
:
з
.gradients_1/Decoder/reshape_image_grad/ReshapeReshape-gradients_1/Sub_grad/tuple/control_dependency,gradients_1/Decoder/reshape_image_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

2gradients_1/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:џџџџџџџџџ
т
.gradients_1/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_1/logistic_loss/Exp_grad/mul2gradients_1/logistic_loss/Select_1_grad/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
ф
0gradients_1/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_1/logistic_loss/Select_1_grad/zeros_like&gradients_1/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Є
8gradients_1/logistic_loss/Select_1_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss/Select_1_grad/Select1^gradients_1/logistic_loss/Select_1_grad/Select_1
Ќ
@gradients_1/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss/Select_1_grad/Select9^gradients_1/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@gradients_1/logistic_loss/Select_1_grad/Select
В
Bgradients_1/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss/Select_1_grad/Select_19^gradients_1/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*C
_class9
75loc:@gradients_1/logistic_loss/Select_1_grad/Select_1
Џ
,gradients_1/Decoder/last_layer_grad/TanhGradTanhGradDecoder/last_layer.gradients_1/Decoder/reshape_image_grad/Reshape*
T0*(
_output_shapes
:џџџџџџџџџ
Ё
&gradients_1/logistic_loss/Neg_grad/NegNeg@gradients_1/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
2gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/Decoder/last_layer_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ѓ
7gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad-^gradients_1/Decoder/last_layer_grad/TanhGrad
Ї
?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_1/Decoder/last_layer_grad/TanhGrad8^gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/Decoder/last_layer_grad/TanhGrad*(
_output_shapes
:џџџџџџџџџ
Ј
Agradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad8^gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ы
,gradients_1/Decoder/dense/MatMul_grad/MatMulMatMul?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependencyDecoder/dense/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ы
.gradients_1/Decoder/dense/MatMul_grad/MatMul_1MatMulDecoder/second_layer/leaky_relu?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(

6gradients_1/Decoder/dense/MatMul_grad/tuple/group_depsNoOp-^gradients_1/Decoder/dense/MatMul_grad/MatMul/^gradients_1/Decoder/dense/MatMul_grad/MatMul_1
Ѕ
>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/Decoder/dense/MatMul_grad/MatMul7^gradients_1/Decoder/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/Decoder/dense/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
Ѓ
@gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/Decoder/dense/MatMul_grad/MatMul_17^gradients_1/Decoder/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/Decoder/dense/MatMul_grad/MatMul_1* 
_output_shapes
:

§
gradients_1/AddNAddN>gradients_1/logistic_loss/Select_grad/tuple/control_dependency;gradients_1/logistic_loss/mul_grad/tuple/control_dependencyBgradients_1/logistic_loss/Select_1_grad/tuple/control_dependency_1&gradients_1/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:џџџџџџџџџ

7gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes
:

<gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN8^gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad

Dgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN=^gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Л
Fgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity7gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad=^gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad

6gradients_1/Decoder/second_layer/leaky_relu_grad/ShapeShape#Decoder/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
А
8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1Shape8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Ж
8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_2Shape>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

<gradients_1/Decoder/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ћ
6gradients_1/Decoder/second_layer/leaky_relu_grad/zerosFill8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_2<gradients_1/Decoder/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
п
=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Decoder/second_layer/leaky_relu/mul8Decoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0

Fgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Decoder/second_layer/leaky_relu_grad/Shape8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
7gradients_1/Decoder/second_layer/leaky_relu_grad/SelectSelect=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqual>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency6gradients_1/Decoder/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
­
9gradients_1/Decoder/second_layer/leaky_relu_grad/Select_1Select=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqual6gradients_1/Decoder/second_layer/leaky_relu_grad/zeros>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
ќ
4gradients_1/Decoder/second_layer/leaky_relu_grad/SumSum7gradients_1/Decoder/second_layer/leaky_relu_grad/SelectFgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ђ
8gradients_1/Decoder/second_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Decoder/second_layer/leaky_relu_grad/Sum6gradients_1/Decoder/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

6gradients_1/Decoder/second_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Decoder/second_layer/leaky_relu_grad/Select_1Hgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ј
:gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Decoder/second_layer/leaky_relu_grad/Sum_18gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
С
Agradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape;^gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1
г
Igradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Decoder/second_layer/leaky_relu_grad/ReshapeB^gradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
й
Kgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1B^gradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
њ
1gradients_1/Discriminator/prob/MatMul_grad/MatMulMatMulDgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
њ
3gradients_1/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluDgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
­
;gradients_1/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp2^gradients_1/Discriminator/prob/MatMul_grad/MatMul4^gradients_1/Discriminator/prob/MatMul_grad/MatMul_1
Й
Cgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity1gradients_1/Discriminator/prob/MatMul_grad/MatMul<^gradients_1/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/Discriminator/prob/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
Ж
Egradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity3gradients_1/Discriminator/prob/MatMul_grad/MatMul_1<^gradients_1/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	
}
:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Д
<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1Shape8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

Jgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ї
8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/MulMulIgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency8Decoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0

8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/SumSum8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/MulJgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ь
<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ц
:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Mul_1Mul%Decoder/second_layer/leaky_relu/alphaIgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

>gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Э
Egradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1
б
Mgradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*O
_classE
CAloc:@gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape
щ
Ogradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Ѕ
<gradients_1/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
А
>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
С
>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Bgradients_1/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

<gradients_1/Discriminator/second_layer/leaky_relu_grad/zerosFill>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_2Bgradients_1/Discriminator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
х
Cgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
 
Lgradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Т
=gradients_1/Discriminator/second_layer/leaky_relu_grad/SelectSelectCgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqualCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency<gradients_1/Discriminator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
Ф
?gradients_1/Discriminator/second_layer/leaky_relu_grad/Select_1SelectCgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqual<gradients_1/Discriminator/second_layer/leaky_relu_grad/zerosCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

:gradients_1/Discriminator/second_layer/leaky_relu_grad/SumSum=gradients_1/Discriminator/second_layer/leaky_relu_grad/SelectLgradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape:gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum<gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

<gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum?gradients_1/Discriminator/second_layer/leaky_relu_grad/Select_1Ngradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape<gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum_1>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Ggradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp?^gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeA^gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1
ы
Ogradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity>gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeH^gradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1H^gradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1
У
gradients_1/AddN_1AddNKgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Ч
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Decoder/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:

Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
й
_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_1_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_1agradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
Sgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

Zgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
З
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
А
dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1

@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Д
Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ќ
Pgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ShapeBgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
§
>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/MulMulOgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/SumSum>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/MulPgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ў
Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ј
@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaOgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Rgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Dgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
п
Kgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
щ
Sgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeL^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape

Ugradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1L^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Л
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Decoder/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:

Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
й
_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѓ
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Decoder/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
Ф
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Decoder/second_layer/fully_connected/BiasAddbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Ъ
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
Sgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

Zgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
З
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
А
dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
е
gradients_1/AddN_2AddNQgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Ugradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Џ
Ogradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:
У
Tgradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2P^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
л
\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2U^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradU^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
љ
Igradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Ngradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
А
Vgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ

Xgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Т
Igradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMul\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Њ
Kgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_relu\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ѕ
Sgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulL^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1

[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulT^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*\
_classR
PNloc:@gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul

]gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1T^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

А
Cgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Decoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

Egradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1MatMulDecoder/first_layer/leaky_reluVgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
у
Mgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1

Ugradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*V
_classL
JHloc:@gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul
џ
Wgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

Ѓ
;gradients_1/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ў
=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
и
=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_2Shape[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Agradients_1/Discriminator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

;gradients_1/Discriminator/first_layer/leaky_relu_grad/zerosFill=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_2Agradients_1/Discriminator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
т
Bgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

Kgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
з
<gradients_1/Discriminator/first_layer/leaky_relu_grad/SelectSelectBgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqual[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency;gradients_1/Discriminator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
й
>gradients_1/Discriminator/first_layer/leaky_relu_grad/Select_1SelectBgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqual;gradients_1/Discriminator/first_layer/leaky_relu_grad/zeros[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

9gradients_1/Discriminator/first_layer/leaky_relu_grad/SumSum<gradients_1/Discriminator/first_layer/leaky_relu_grad/SelectKgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

=gradients_1/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape9gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum;gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

;gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum>gradients_1/Discriminator/first_layer/leaky_relu_grad/Select_1Mgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

?gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape;gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum_1=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
а
Fgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp>^gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape@^gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1
ч
Ngradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity=gradients_1/Discriminator/first_layer/leaky_relu_grad/ReshapeG^gradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
э
Pgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity?gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1G^gradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

5gradients_1/Decoder/first_layer/leaky_relu_grad/ShapeShape"Decoder/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ђ
7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1Shape+Decoder/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ь
7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

;gradients_1/Decoder/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
5gradients_1/Decoder/first_layer/leaky_relu_grad/zerosFill7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_2;gradients_1/Decoder/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
а
<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual"Decoder/first_layer/leaky_relu/mul+Decoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

Egradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/Decoder/first_layer/leaky_relu_grad/Shape7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
П
6gradients_1/Decoder/first_layer/leaky_relu_grad/SelectSelect<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqualUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency5gradients_1/Decoder/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
С
8gradients_1/Decoder/first_layer/leaky_relu_grad/Select_1Select<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqual5gradients_1/Decoder/first_layer/leaky_relu_grad/zerosUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
љ
3gradients_1/Decoder/first_layer/leaky_relu_grad/SumSum6gradients_1/Decoder/first_layer/leaky_relu_grad/SelectEgradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
я
7gradients_1/Decoder/first_layer/leaky_relu_grad/ReshapeReshape3gradients_1/Decoder/first_layer/leaky_relu_grad/Sum5gradients_1/Decoder/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
џ
5gradients_1/Decoder/first_layer/leaky_relu_grad/Sum_1Sum8gradients_1/Decoder/first_layer/leaky_relu_grad/Select_1Ggradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ѕ
9gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1Reshape5gradients_1/Decoder/first_layer/leaky_relu_grad/Sum_17gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
О
@gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_depsNoOp8^gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape:^gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1
Я
Hgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity7gradients_1/Decoder/first_layer/leaky_relu_grad/ReshapeA^gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
е
Jgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity9gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1A^gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
В
Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Љ
Ogradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ShapeAgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
њ
=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/MulMulNgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/SumSum=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/MulOgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ћ
Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ѕ
?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaNgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Qgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Cgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
м
Jgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeD^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
х
Rgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeK^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
§
Tgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1K^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
|
9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
І
;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1Shape+Decoder/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Igradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ш
7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/MulMulHgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency+Decoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/SumSum7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/MulIgradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
щ
;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ReshapeReshape7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
у
9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Mul_1Mul$Decoder/first_layer/leaky_relu/alphaHgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum_1Sum9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Mul_1Kgradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

=gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1Reshape9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum_1;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ъ
Dgradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp<^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape>^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1
Э
Lgradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
х
Ngradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity=gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1E^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
в
gradients_1/AddN_3AddNPgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Tgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1
Ў
Ngradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_3*
data_formatNHWC*
_output_shapes	
:*
T0
С
Sgradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_3O^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
и
[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_3T^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

]gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradT^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Р
gradients_1/AddN_4AddNJgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Ngradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Ј
Hgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:
Е
Mgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_4I^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ц
Ugradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_4N^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

Wgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradN^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
О
Hgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMul[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(

Jgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_code[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d*
transpose_a(*
transpose_b( *
T0
ђ
Rgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulK^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1

Zgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulS^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџd*
T0*[
_classQ
OMloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul

\gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1S^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d
Ќ
Bgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulMatMulUgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency/Decoder/first_layer/fully_connected/kernel/read*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(*
T0

Dgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_codeUgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	d*
transpose_a(
р
Lgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpC^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulE^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1
ќ
Tgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityBgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulM^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџd*
T0*U
_classK
IGloc:@gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul
њ
Vgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityDgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1M^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d
ф
gradients_1/AddN_5AddNZgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyTgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*[
_classQ
OMloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*
N*'
_output_shapes
:џџџџџџџџџd

1gradients_1/Encoder/encoder_code_grad/SigmoidGradSigmoidGradEncoder/encoder_codegradients_1/AddN_5*
T0*'
_output_shapes
:џџџџџџџџџd
t
"gradients_1/Encoder/Add_grad/ShapeShapeEncoder/logvar_std*
T0*
out_type0*
_output_shapes
:
~
$gradients_1/Encoder/Add_grad/Shape_1ShapeEncoder/encoder_mu/BiasAdd*
T0*
out_type0*
_output_shapes
:
в
2gradients_1/Encoder/Add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_1/Encoder/Add_grad/Shape$gradients_1/Encoder/Add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ю
 gradients_1/Encoder/Add_grad/SumSum1gradients_1/Encoder/encoder_code_grad/SigmoidGrad2gradients_1/Encoder/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Е
$gradients_1/Encoder/Add_grad/ReshapeReshape gradients_1/Encoder/Add_grad/Sum"gradients_1/Encoder/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџd
в
"gradients_1/Encoder/Add_grad/Sum_1Sum1gradients_1/Encoder/encoder_code_grad/SigmoidGrad4gradients_1/Encoder/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
&gradients_1/Encoder/Add_grad/Reshape_1Reshape"gradients_1/Encoder/Add_grad/Sum_1$gradients_1/Encoder/Add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџd

-gradients_1/Encoder/Add_grad/tuple/group_depsNoOp%^gradients_1/Encoder/Add_grad/Reshape'^gradients_1/Encoder/Add_grad/Reshape_1

5gradients_1/Encoder/Add_grad/tuple/control_dependencyIdentity$gradients_1/Encoder/Add_grad/Reshape.^gradients_1/Encoder/Add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients_1/Encoder/Add_grad/Reshape*'
_output_shapes
:џџџџџџџџџd

7gradients_1/Encoder/Add_grad/tuple/control_dependency_1Identity&gradients_1/Encoder/Add_grad/Reshape_1.^gradients_1/Encoder/Add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/Encoder/Add_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџd
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
ч
9gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients_1/Encoder/logvar_std_grad/Shape+gradients_1/Encoder/logvar_std_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Є
'gradients_1/Encoder/logvar_std_grad/MulMul5gradients_1/Encoder/Add_grad/tuple/control_dependencyEncoder/Exp*
T0*'
_output_shapes
:џџџџџџџџџd
в
'gradients_1/Encoder/logvar_std_grad/SumSum'gradients_1/Encoder/logvar_std_grad/Mul9gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
+gradients_1/Encoder/logvar_std_grad/ReshapeReshape'gradients_1/Encoder/logvar_std_grad/Sum)gradients_1/Encoder/logvar_std_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
А
)gradients_1/Encoder/logvar_std_grad/Mul_1MulEncoder/random_normal5gradients_1/Encoder/Add_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџd
и
)gradients_1/Encoder/logvar_std_grad/Sum_1Sum)gradients_1/Encoder/logvar_std_grad/Mul_1;gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
а
-gradients_1/Encoder/logvar_std_grad/Reshape_1Reshape)gradients_1/Encoder/logvar_std_grad/Sum_1+gradients_1/Encoder/logvar_std_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџd

4gradients_1/Encoder/logvar_std_grad/tuple/group_depsNoOp,^gradients_1/Encoder/logvar_std_grad/Reshape.^gradients_1/Encoder/logvar_std_grad/Reshape_1

<gradients_1/Encoder/logvar_std_grad/tuple/control_dependencyIdentity+gradients_1/Encoder/logvar_std_grad/Reshape5^gradients_1/Encoder/logvar_std_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients_1/Encoder/logvar_std_grad/Reshape*
_output_shapes
:d
Є
>gradients_1/Encoder/logvar_std_grad/tuple/control_dependency_1Identity-gradients_1/Encoder/logvar_std_grad/Reshape_15^gradients_1/Encoder/logvar_std_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients_1/Encoder/logvar_std_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџd
Л
7gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/Encoder/Add_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:d*
T0
И
<gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_depsNoOp8^gradients_1/Encoder/Add_grad/tuple/control_dependency_18^gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad
Е
Dgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/Encoder/Add_grad/tuple/control_dependency_1=^gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџd*
T0*9
_class/
-+loc:@gradients_1/Encoder/Add_grad/Reshape_1
Л
Fgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency_1Identity7gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad=^gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_deps*
_output_shapes
:d*
T0*J
_class@
><loc:@gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad
І
 gradients_1/Encoder/Exp_grad/mulMul>gradients_1/Encoder/logvar_std_grad/tuple/control_dependency_1Encoder/Exp*
T0*'
_output_shapes
:џџџџџџџџџd
њ
1gradients_1/Encoder/encoder_mu/MatMul_grad/MatMulMatMulDgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependencyEncoder/encoder_mu/kernel/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
є
3gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1MatMulEncoder/second_layer/leaky_reluDgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d*
transpose_a(*
transpose_b( *
T0
­
;gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_depsNoOp2^gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul4^gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1
Й
Cgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependencyIdentity1gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul<^gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*D
_class:
86loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul
Ж
Egradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependency_1Identity3gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1<^gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1*
_output_shapes
:	d

&gradients_1/Encoder/truediv_grad/ShapeShapeEncoder/encoder_logvar/BiasAdd*
_output_shapes
:*
T0*
out_type0
k
(gradients_1/Encoder/truediv_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
о
6gradients_1/Encoder/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/Encoder/truediv_grad/Shape(gradients_1/Encoder/truediv_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

(gradients_1/Encoder/truediv_grad/RealDivRealDiv gradients_1/Encoder/Exp_grad/mulEncoder/truediv/y*'
_output_shapes
:џџџџџџџџџd*
T0
Э
$gradients_1/Encoder/truediv_grad/SumSum(gradients_1/Encoder/truediv_grad/RealDiv6gradients_1/Encoder/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(gradients_1/Encoder/truediv_grad/ReshapeReshape$gradients_1/Encoder/truediv_grad/Sum&gradients_1/Encoder/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџd
}
$gradients_1/Encoder/truediv_grad/NegNegEncoder/encoder_logvar/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџd
 
*gradients_1/Encoder/truediv_grad/RealDiv_1RealDiv$gradients_1/Encoder/truediv_grad/NegEncoder/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџd
І
*gradients_1/Encoder/truediv_grad/RealDiv_2RealDiv*gradients_1/Encoder/truediv_grad/RealDiv_1Encoder/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџd
Ћ
$gradients_1/Encoder/truediv_grad/mulMul gradients_1/Encoder/Exp_grad/mul*gradients_1/Encoder/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:џџџџџџџџџd
Э
&gradients_1/Encoder/truediv_grad/Sum_1Sum$gradients_1/Encoder/truediv_grad/mul8gradients_1/Encoder/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
*gradients_1/Encoder/truediv_grad/Reshape_1Reshape&gradients_1/Encoder/truediv_grad/Sum_1(gradients_1/Encoder/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

1gradients_1/Encoder/truediv_grad/tuple/group_depsNoOp)^gradients_1/Encoder/truediv_grad/Reshape+^gradients_1/Encoder/truediv_grad/Reshape_1

9gradients_1/Encoder/truediv_grad/tuple/control_dependencyIdentity(gradients_1/Encoder/truediv_grad/Reshape2^gradients_1/Encoder/truediv_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/Encoder/truediv_grad/Reshape*'
_output_shapes
:џџџџџџџџџd

;gradients_1/Encoder/truediv_grad/tuple/control_dependency_1Identity*gradients_1/Encoder/truediv_grad/Reshape_12^gradients_1/Encoder/truediv_grad/tuple/group_deps*
_output_shapes
: *
T0*=
_class3
1/loc:@gradients_1/Encoder/truediv_grad/Reshape_1
С
;gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients_1/Encoder/truediv_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:d
Т
@gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_depsNoOp<^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGrad:^gradients_1/Encoder/truediv_grad/tuple/control_dependency
С
Hgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependencyIdentity9gradients_1/Encoder/truediv_grad/tuple/control_dependencyA^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/Encoder/truediv_grad/Reshape*'
_output_shapes
:џџџџџџџџџd
Ы
Jgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGradA^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_deps*
_output_shapes
:d*
T0*N
_classD
B@loc:@gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGrad

5gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMulMatMulHgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency"Encoder/encoder_logvar/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ќ
7gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1MatMulEncoder/second_layer/leaky_reluHgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d*
transpose_a(*
transpose_b( 
Й
?gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_depsNoOp6^gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul8^gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1
Щ
Ggradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependencyIdentity5gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul@^gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
Ц
Igradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency_1Identity7gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1@^gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1*
_output_shapes
:	d
Њ
gradients_1/AddN_6AddNCgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependencyGgradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency*
T0*D
_class:
86loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul*
N*(
_output_shapes
:џџџџџџџџџ

6gradients_1/Encoder/second_layer/leaky_relu_grad/ShapeShape#Encoder/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
А
8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1Shape8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_2Shapegradients_1/AddN_6*
T0*
out_type0*
_output_shapes
:

<gradients_1/Encoder/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ћ
6gradients_1/Encoder/second_layer/leaky_relu_grad/zerosFill8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_2<gradients_1/Encoder/second_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
п
=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Encoder/second_layer/leaky_relu/mul8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

Fgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Encoder/second_layer/leaky_relu_grad/Shape8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
џ
7gradients_1/Encoder/second_layer/leaky_relu_grad/SelectSelect=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqualgradients_1/AddN_66gradients_1/Encoder/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients_1/Encoder/second_layer/leaky_relu_grad/Select_1Select=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqual6gradients_1/Encoder/second_layer/leaky_relu_grad/zerosgradients_1/AddN_6*
T0*(
_output_shapes
:џџџџџџџџџ
ќ
4gradients_1/Encoder/second_layer/leaky_relu_grad/SumSum7gradients_1/Encoder/second_layer/leaky_relu_grad/SelectFgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ђ
8gradients_1/Encoder/second_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Encoder/second_layer/leaky_relu_grad/Sum6gradients_1/Encoder/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

6gradients_1/Encoder/second_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Encoder/second_layer/leaky_relu_grad/Select_1Hgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ј
:gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Encoder/second_layer/leaky_relu_grad/Sum_18gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
С
Agradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape;^gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1
г
Igradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Encoder/second_layer/leaky_relu_grad/ReshapeB^gradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
й
Kgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1B^gradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
}
:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Д
<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1Shape8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

Jgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ї
8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/MulMulIgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/SumSum8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/MulJgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ь
<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ц
:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Mul_1Mul%Encoder/second_layer/leaky_relu/alphaIgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Э
Egradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1
б
Mgradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
щ
Ogradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1
У
gradients_1/AddN_7AddNKgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Ч
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Encoder/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:

Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
й
_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_7_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_7agradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
Sgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
З
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape
А
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Л
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Encoder/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:

Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
й
_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѓ
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Encoder/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
Ф
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Н
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Encoder/second_layer/fully_connected/BiasAddbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Ъ
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
Sgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
З
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
А
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
о
Kgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Xgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg
Л
`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1

bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*^
_classT
RPloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg
љ
Igradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:*
T0

Ngradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
А
Vgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape

Xgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*\
_classR
PNloc:@gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad

Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Encoder/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:*
T0

Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Encoder/second_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:

Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Ђ
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
Ј
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
А
Cgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Encoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

Egradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/first_layer/leaky_reluVgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
у
Mgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1

Ugradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
џ
Wgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*X
_classN
LJloc:@gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1
§
gradients_1/AddN_8AddNdgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
С
Kgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_83Encoder/second_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
Ш
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_88Encoder/second_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:
ў
Xgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1

`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:
 
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:

5gradients_1/Encoder/first_layer/leaky_relu_grad/ShapeShape"Encoder/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ђ
7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1Shape+Encoder/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ь
7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

;gradients_1/Encoder/first_layer/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
ј
5gradients_1/Encoder/first_layer/leaky_relu_grad/zerosFill7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_2;gradients_1/Encoder/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
а
<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual"Encoder/first_layer/leaky_relu/mul+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

Egradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/Encoder/first_layer/leaky_relu_grad/Shape7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
П
6gradients_1/Encoder/first_layer/leaky_relu_grad/SelectSelect<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqualUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency5gradients_1/Encoder/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
С
8gradients_1/Encoder/first_layer/leaky_relu_grad/Select_1Select<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqual5gradients_1/Encoder/first_layer/leaky_relu_grad/zerosUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
љ
3gradients_1/Encoder/first_layer/leaky_relu_grad/SumSum6gradients_1/Encoder/first_layer/leaky_relu_grad/SelectEgradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
я
7gradients_1/Encoder/first_layer/leaky_relu_grad/ReshapeReshape3gradients_1/Encoder/first_layer/leaky_relu_grad/Sum5gradients_1/Encoder/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
џ
5gradients_1/Encoder/first_layer/leaky_relu_grad/Sum_1Sum8gradients_1/Encoder/first_layer/leaky_relu_grad/Select_1Ggradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ѕ
9gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1Reshape5gradients_1/Encoder/first_layer/leaky_relu_grad/Sum_17gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
О
@gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_depsNoOp8^gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape:^gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1
Я
Hgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity7gradients_1/Encoder/first_layer/leaky_relu_grad/ReshapeA^gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
е
Jgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity9gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1A^gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
|
9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
І
;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1Shape+Encoder/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Igradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ш
7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/MulMulHgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency+Encoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/SumSum7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/MulIgradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
щ
;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ReshapeReshape7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
у
9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Mul_1Mul$Encoder/first_layer/leaky_relu/alphaHgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum_1Sum9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Mul_1Kgradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

=gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1Reshape9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum_1;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ъ
Dgradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp<^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape>^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1
Э
Lgradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
х
Ngradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity=gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1E^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Р
gradients_1/AddN_9AddNJgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Ngradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1
Ј
Hgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:
Е
Mgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_9I^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ц
Ugradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_9N^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

Wgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradN^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
­
Bgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulMatMulUgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency/Encoder/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

Dgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/ReshapeUgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
р
Lgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpC^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulE^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1
§
Tgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityBgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulM^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
ћ
Vgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityDgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1M^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:


beta1_power_1/initial_valueConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ё
beta1_power_1
VariableV2*
shared_name *.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
Ф
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
: 
~
beta1_power_1/readIdentitybeta1_power_1*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
: 

beta2_power_1/initial_valueConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Ё
beta2_power_1
VariableV2*.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Ф
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
: 
~
beta2_power_1/readIdentitybeta2_power_1*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
: *
T0
ы
[Encoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
е
QEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
џ
KEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill[Encoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorQEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
*
T0
№
9Encoder/first_layer/fully_connected/kernel/generator_opti
VariableV2*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
х
@Encoder/first_layer/fully_connected/kernel/generator_opti/AssignAssign9Encoder/first_layer/fully_connected/kernel/generator_optiKEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

я
>Encoder/first_layer/fully_connected/kernel/generator_opti/readIdentity9Encoder/first_layer/fully_connected/kernel/generator_opti* 
_output_shapes
:
*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel
э
]Encoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
з
SEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*
_output_shapes
: *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0

MEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill]Encoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorSEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
*
T0
ђ
;Encoder/first_layer/fully_connected/kernel/generator_opti_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel
ы
BEncoder/first_layer/fully_connected/kernel/generator_opti_1/AssignAssign;Encoder/first_layer/fully_connected/kernel/generator_opti_1MEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ѓ
@Encoder/first_layer/fully_connected/kernel/generator_opti_1/readIdentity;Encoder/first_layer/fully_connected/kernel/generator_opti_1* 
_output_shapes
:
*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel
е
IEncoder/first_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
т
7Encoder/first_layer/fully_connected/bias/generator_opti
VariableV2*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
и
>Encoder/first_layer/fully_connected/bias/generator_opti/AssignAssign7Encoder/first_layer/fully_connected/bias/generator_optiIEncoder/first_layer/fully_connected/bias/generator_opti/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ф
<Encoder/first_layer/fully_connected/bias/generator_opti/readIdentity7Encoder/first_layer/fully_connected/bias/generator_opti*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:*
T0
з
KEncoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ф
9Encoder/first_layer/fully_connected/bias/generator_opti_1
VariableV2*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
	container *
shape:*
dtype0
о
@Encoder/first_layer/fully_connected/bias/generator_opti_1/AssignAssign9Encoder/first_layer/fully_connected/bias/generator_opti_1KEncoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ш
>Encoder/first_layer/fully_connected/bias/generator_opti_1/readIdentity9Encoder/first_layer/fully_connected/bias/generator_opti_1*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:*
T0
э
\Encoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
з
REncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

LEncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill\Encoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorREncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ђ
:Encoder/second_layer/fully_connected/kernel/generator_opti
VariableV2*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

щ
AEncoder/second_layer/fully_connected/kernel/generator_opti/AssignAssign:Encoder/second_layer/fully_connected/kernel/generator_optiLEncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ђ
?Encoder/second_layer/fully_connected/kernel/generator_opti/readIdentity:Encoder/second_layer/fully_connected/kernel/generator_opti*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
я
^Encoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
й
TEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*
_output_shapes
: *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0

NEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill^Encoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorTEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

є
<Encoder/second_layer/fully_connected/kernel/generator_opti_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:

я
CEncoder/second_layer/fully_connected/kernel/generator_opti_1/AssignAssign<Encoder/second_layer/fully_connected/kernel/generator_opti_1NEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

і
AEncoder/second_layer/fully_connected/kernel/generator_opti_1/readIdentity<Encoder/second_layer/fully_connected/kernel/generator_opti_1* 
_output_shapes
:
*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
з
JEncoder/second_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ф
8Encoder/second_layer/fully_connected/bias/generator_opti
VariableV2*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container *
shape:*
dtype0
м
?Encoder/second_layer/fully_connected/bias/generator_opti/AssignAssign8Encoder/second_layer/fully_connected/bias/generator_optiJEncoder/second_layer/fully_connected/bias/generator_opti/Initializer/zeros*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ч
=Encoder/second_layer/fully_connected/bias/generator_opti/readIdentity8Encoder/second_layer/fully_connected/bias/generator_opti*
_output_shapes	
:*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias
й
LEncoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ц
:Encoder/second_layer/fully_connected/bias/generator_opti_1
VariableV2*
shared_name *<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
т
AEncoder/second_layer/fully_connected/bias/generator_opti_1/AssignAssign:Encoder/second_layer/fully_connected/bias/generator_opti_1LEncoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(
ы
?Encoder/second_layer/fully_connected/bias/generator_opti_1/readIdentity:Encoder/second_layer/fully_connected/bias/generator_opti_1*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
_output_shapes	
:*
T0
с
OEncoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zerosConst*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ю
=Encoder/second_layer/batch_normalization/gamma/generator_opti
VariableV2*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
№
DEncoder/second_layer/batch_normalization/gamma/generator_opti/AssignAssign=Encoder/second_layer/batch_normalization/gamma/generator_optiOEncoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
і
BEncoder/second_layer/batch_normalization/gamma/generator_opti/readIdentity=Encoder/second_layer/batch_normalization/gamma/generator_opti*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:
у
QEncoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zerosConst*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
№
?Encoder/second_layer/batch_normalization/gamma/generator_opti_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container *
shape:
і
FEncoder/second_layer/batch_normalization/gamma/generator_opti_1/AssignAssign?Encoder/second_layer/batch_normalization/gamma/generator_opti_1QEncoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zeros*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
њ
DEncoder/second_layer/batch_normalization/gamma/generator_opti_1/readIdentity?Encoder/second_layer/batch_normalization/gamma/generator_opti_1*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:
п
NEncoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zerosConst*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ь
<Encoder/second_layer/batch_normalization/beta/generator_opti
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
	container *
shape:
ь
CEncoder/second_layer/batch_normalization/beta/generator_opti/AssignAssign<Encoder/second_layer/batch_normalization/beta/generator_optiNEncoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
ѓ
AEncoder/second_layer/batch_normalization/beta/generator_opti/readIdentity<Encoder/second_layer/batch_normalization/beta/generator_opti*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:
с
PEncoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zerosConst*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ю
>Encoder/second_layer/batch_normalization/beta/generator_opti_1
VariableV2*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ђ
EEncoder/second_layer/batch_normalization/beta/generator_opti_1/AssignAssign>Encoder/second_layer/batch_normalization/beta/generator_opti_1PEncoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zeros*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ї
CEncoder/second_layer/batch_normalization/beta/generator_opti_1/readIdentity>Encoder/second_layer/batch_normalization/beta/generator_opti_1*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:
Щ
JEncoder/encoder_mu/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
Г
@Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros/ConstConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
К
:Encoder/encoder_mu/kernel/generator_opti/Initializer/zerosFillJEncoder/encoder_mu/kernel/generator_opti/Initializer/zeros/shape_as_tensor@Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros/Const*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*

index_type0*
_output_shapes
:	d
Ь
(Encoder/encoder_mu/kernel/generator_opti
VariableV2*,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name 
 
/Encoder/encoder_mu/kernel/generator_opti/AssignAssign(Encoder/encoder_mu/kernel/generator_opti:Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0
Л
-Encoder/encoder_mu/kernel/generator_opti/readIdentity(Encoder/encoder_mu/kernel/generator_opti*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	d
Ы
LEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
Е
BEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/ConstConst*
_output_shapes
: *,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *    *
dtype0
Р
<Encoder/encoder_mu/kernel/generator_opti_1/Initializer/zerosFillLEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorBEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/Const*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*

index_type0*
_output_shapes
:	d
Ю
*Encoder/encoder_mu/kernel/generator_opti_1
VariableV2*
shared_name *,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d
І
1Encoder/encoder_mu/kernel/generator_opti_1/AssignAssign*Encoder/encoder_mu/kernel/generator_opti_1<Encoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros*
_output_shapes
:	d*
use_locking(*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(
П
/Encoder/encoder_mu/kernel/generator_opti_1/readIdentity*Encoder/encoder_mu/kernel/generator_opti_1*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	d
Б
8Encoder/encoder_mu/bias/generator_opti/Initializer/zerosConst**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0*
_output_shapes
:d
О
&Encoder/encoder_mu/bias/generator_opti
VariableV2*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name **
_class 
loc:@Encoder/encoder_mu/bias

-Encoder/encoder_mu/bias/generator_opti/AssignAssign&Encoder/encoder_mu/bias/generator_opti8Encoder/encoder_mu/bias/generator_opti/Initializer/zeros**
_class 
loc:@Encoder/encoder_mu/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
А
+Encoder/encoder_mu/bias/generator_opti/readIdentity&Encoder/encoder_mu/bias/generator_opti*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d
Г
:Encoder/encoder_mu/bias/generator_opti_1/Initializer/zerosConst**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0*
_output_shapes
:d
Р
(Encoder/encoder_mu/bias/generator_opti_1
VariableV2*
shared_name **
_class 
loc:@Encoder/encoder_mu/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d

/Encoder/encoder_mu/bias/generator_opti_1/AssignAssign(Encoder/encoder_mu/bias/generator_opti_1:Encoder/encoder_mu/bias/generator_opti_1/Initializer/zeros*
_output_shapes
:d*
use_locking(*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
validate_shape(
Д
-Encoder/encoder_mu/bias/generator_opti_1/readIdentity(Encoder/encoder_mu/bias/generator_opti_1*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d
б
NEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0
Л
DEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/ConstConst*
_output_shapes
: *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *    *
dtype0
Ъ
>Encoder/encoder_logvar/kernel/generator_opti/Initializer/zerosFillNEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/shape_as_tensorDEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/Const*
_output_shapes
:	d*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*

index_type0
д
,Encoder/encoder_logvar/kernel/generator_opti
VariableV2*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d
А
3Encoder/encoder_logvar/kernel/generator_opti/AssignAssign,Encoder/encoder_logvar/kernel/generator_opti>Encoder/encoder_logvar/kernel/generator_opti/Initializer/zeros*
_output_shapes
:	d*
use_locking(*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(
Ч
1Encoder/encoder_logvar/kernel/generator_opti/readIdentity,Encoder/encoder_logvar/kernel/generator_opti*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	d*
T0
г
PEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
Н
FEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/ConstConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
а
@Encoder/encoder_logvar/kernel/generator_opti_1/Initializer/zerosFillPEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorFEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/Const*
_output_shapes
:	d*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*

index_type0
ж
.Encoder/encoder_logvar/kernel/generator_opti_1
VariableV2*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d
Ж
5Encoder/encoder_logvar/kernel/generator_opti_1/AssignAssign.Encoder/encoder_logvar/kernel/generator_opti_1@Encoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros*
_output_shapes
:	d*
use_locking(*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(
Ы
3Encoder/encoder_logvar/kernel/generator_opti_1/readIdentity.Encoder/encoder_logvar/kernel/generator_opti_1*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	d
Й
<Encoder/encoder_logvar/bias/generator_opti/Initializer/zerosConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0*
_output_shapes
:d
Ц
*Encoder/encoder_logvar/bias/generator_opti
VariableV2*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name *.
_class$
" loc:@Encoder/encoder_logvar/bias
Ѓ
1Encoder/encoder_logvar/bias/generator_opti/AssignAssign*Encoder/encoder_logvar/bias/generator_opti<Encoder/encoder_logvar/bias/generator_opti/Initializer/zeros*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
М
/Encoder/encoder_logvar/bias/generator_opti/readIdentity*Encoder/encoder_logvar/bias/generator_opti*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
:d
Л
>Encoder/encoder_logvar/bias/generator_opti_1/Initializer/zerosConst*
_output_shapes
:d*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0
Ш
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
Љ
3Encoder/encoder_logvar/bias/generator_opti_1/AssignAssign,Encoder/encoder_logvar/bias/generator_opti_1>Encoder/encoder_logvar/bias/generator_opti_1/Initializer/zeros*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
Р
1Encoder/encoder_logvar/bias/generator_opti_1/readIdentity,Encoder/encoder_logvar/bias/generator_opti_1*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
:d
a
generator_opti/learning_rateConst*
_output_shapes
: *
valueB
 *ЗQ9*
dtype0
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
 *wО?*
dtype0*
_output_shapes
: 
[
generator_opti/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
ю
Jgenerator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam*Encoder/first_layer/fully_connected/kernel9Encoder/first_layer/fully_connected/kernel/generator_opti;Encoder/first_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonVgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
р
Hgenerator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam(Encoder/first_layer/fully_connected/bias7Encoder/first_layer/fully_connected/bias/generator_opti9Encoder/first_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
use_nesterov( 
є
Kgenerator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Encoder/second_layer/fully_connected/kernel:Encoder/second_layer/fully_connected/kernel/generator_opti<Encoder/second_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
ц
Igenerator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Encoder/second_layer/fully_connected/bias8Encoder/second_layer/fully_connected/bias/generator_opti:Encoder/second_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonXgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:

Ngenerator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Encoder/second_layer/batch_normalization/gamma=Encoder/second_layer/batch_normalization/gamma/generator_opti?Encoder/second_layer/batch_normalization/gamma/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
use_nesterov( 

Mgenerator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Encoder/second_layer/batch_normalization/beta<Encoder/second_layer/batch_normalization/beta/generator_opti>Encoder/second_layer/batch_normalization/beta/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilon`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0

9generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdam	ApplyAdamEncoder/encoder_mu/kernel(Encoder/encoder_mu/kernel/generator_opti*Encoder/encoder_mu/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonEgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
use_nesterov( *
_output_shapes
:	d
љ
7generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam	ApplyAdamEncoder/encoder_mu/bias&Encoder/encoder_mu/bias/generator_opti(Encoder/encoder_mu/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonFgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@Encoder/encoder_mu/bias*
use_nesterov( *
_output_shapes
:d

=generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam	ApplyAdamEncoder/encoder_logvar/kernel,Encoder/encoder_logvar/kernel/generator_opti.Encoder/encoder_logvar/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonIgradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
use_nesterov( *
_output_shapes
:	d

;generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam	ApplyAdamEncoder/encoder_logvar/bias*Encoder/encoder_logvar/bias/generator_opti,Encoder/encoder_logvar/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonJgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
use_nesterov( *
_output_shapes
:d
л
generator_opti/mulMulbeta1_power_1/readgenerator_opti/beta1<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
: 
М
generator_opti/AssignAssignbeta1_power_1generator_opti/mul*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
н
generator_opti/mul_1Mulbeta2_power_1/readgenerator_opti/beta2<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias
Р
generator_opti/Assign_1Assignbeta2_power_1generator_opti/mul_1*
use_locking( *
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
: 

generator_optiNoOp^generator_opti/Assign^generator_opti/Assign_1<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam
i
Merge/MergeSummaryMergeSummarygenerator_loss_1discriminator_loss*
N*
_output_shapes
: "r^L[!     RFh^	Ќj.џжAJУ
З
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
2	
ю
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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

2	

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
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
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

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
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.12.02v1.12.0-0-ga6d8ffae09

Encoder/real_inPlaceholder*/
_output_shapes
:џџџџџџџџџ*$
shape:џџџџџџџџџ*
dtype0
f
Encoder/Reshape/shapeConst*
valueB"џџџџ  *
dtype0*
_output_shapes
:

Encoder/ReshapeReshapeEncoder/real_inEncoder/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
л
KEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0
Э
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *HYН*
dtype0
Э
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
С
SEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformKEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ц
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
_output_shapes
: 
к
IEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulSEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:

Ь
EEncoder/first_layer/fully_connected/kernel/Initializer/random_uniformAddIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulIEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
с
*Encoder/first_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container *
shape:

С
1Encoder/first_layer/fully_connected/kernel/AssignAssign*Encoder/first_layer/fully_connected/kernelEEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

б
/Encoder/first_layer/fully_connected/kernel/readIdentity*Encoder/first_layer/fully_connected/kernel*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:

Ц
:Encoder/first_layer/fully_connected/bias/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
г
(Encoder/first_layer/fully_connected/bias
VariableV2*
shared_name *;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ћ
/Encoder/first_layer/fully_connected/bias/AssignAssign(Encoder/first_layer/fully_connected/bias:Encoder/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ц
-Encoder/first_layer/fully_connected/bias/readIdentity(Encoder/first_layer/fully_connected/bias*
_output_shapes	
:*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias
Я
*Encoder/first_layer/fully_connected/MatMulMatMulEncoder/Reshape/Encoder/first_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
л
+Encoder/first_layer/fully_connected/BiasAddBiasAdd*Encoder/first_layer/fully_connected/MatMul-Encoder/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
i
$Encoder/first_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Џ
"Encoder/first_layer/leaky_relu/mulMul$Encoder/first_layer/leaky_relu/alpha+Encoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
­
Encoder/first_layer/leaky_reluMaximum"Encoder/first_layer/leaky_relu/mul+Encoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
н
LEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0
Я
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *qФН*
dtype0*
_output_shapes
: 
Я
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *qФ=*
dtype0*
_output_shapes
: 
Ф
TEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
Ъ
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
о
JEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel
а
FEncoder/second_layer/fully_connected/kernel/Initializer/random_uniformAddJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulJEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:

у
+Encoder/second_layer/fully_connected/kernel
VariableV2*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Х
2Encoder/second_layer/fully_connected/kernel/AssignAssign+Encoder/second_layer/fully_connected/kernelFEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(
д
0Encoder/second_layer/fully_connected/kernel/readIdentity+Encoder/second_layer/fully_connected/kernel*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
Ш
;Encoder/second_layer/fully_connected/bias/Initializer/zerosConst*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
е
)Encoder/second_layer/fully_connected/bias
VariableV2*
shared_name *<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Џ
0Encoder/second_layer/fully_connected/bias/AssignAssign)Encoder/second_layer/fully_connected/bias;Encoder/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Щ
.Encoder/second_layer/fully_connected/bias/readIdentity)Encoder/second_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
_output_shapes	
:
р
+Encoder/second_layer/fully_connected/MatMulMatMulEncoder/first_layer/leaky_relu0Encoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
о
,Encoder/second_layer/fully_connected/BiasAddBiasAdd+Encoder/second_layer/fully_connected/MatMul.Encoder/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
б
?Encoder/second_layer/batch_normalization/gamma/Initializer/onesConst*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
п
.Encoder/second_layer/batch_normalization/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container *
shape:
Т
5Encoder/second_layer/batch_normalization/gamma/AssignAssign.Encoder/second_layer/batch_normalization/gamma?Encoder/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
и
3Encoder/second_layer/batch_normalization/gamma/readIdentity.Encoder/second_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:
а
?Encoder/second_layer/batch_normalization/beta/Initializer/zerosConst*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
н
-Encoder/second_layer/batch_normalization/beta
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta
П
4Encoder/second_layer/batch_normalization/beta/AssignAssign-Encoder/second_layer/batch_normalization/beta?Encoder/second_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
е
2Encoder/second_layer/batch_normalization/beta/readIdentity-Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta
о
FEncoder/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ы
4Encoder/second_layer/batch_normalization/moving_mean
VariableV2*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
л
;Encoder/second_layer/batch_normalization/moving_mean/AssignAssign4Encoder/second_layer/batch_normalization/moving_meanFEncoder/second_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:
ъ
9Encoder/second_layer/batch_normalization/moving_mean/readIdentity4Encoder/second_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Encoder/second_layer/batch_normalization/moving_mean*
_output_shapes	
:
х
IEncoder/second_layer/batch_normalization/moving_variance/Initializer/onesConst*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ѓ
8Encoder/second_layer/batch_normalization/moving_variance
VariableV2*
shared_name *K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
ъ
?Encoder/second_layer/batch_normalization/moving_variance/AssignAssign8Encoder/second_layer/batch_normalization/moving_varianceIEncoder/second_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:
і
=Encoder/second_layer/batch_normalization/moving_variance/readIdentity8Encoder/second_layer/batch_normalization/moving_variance*
T0*K
_classA
?=loc:@Encoder/second_layer/batch_normalization/moving_variance*
_output_shapes	
:
}
8Encoder/second_layer/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o:*
dtype0
м
6Encoder/second_layer/batch_normalization/batchnorm/addAdd=Encoder/second_layer/batch_normalization/moving_variance/read8Encoder/second_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:*
T0

8Encoder/second_layer/batch_normalization/batchnorm/RsqrtRsqrt6Encoder/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
в
6Encoder/second_layer/batch_normalization/batchnorm/mulMul8Encoder/second_layer/batch_normalization/batchnorm/Rsqrt3Encoder/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
и
8Encoder/second_layer/batch_normalization/batchnorm/mul_1Mul,Encoder/second_layer/fully_connected/BiasAdd6Encoder/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
и
8Encoder/second_layer/batch_normalization/batchnorm/mul_2Mul9Encoder/second_layer/batch_normalization/moving_mean/read6Encoder/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
б
6Encoder/second_layer/batch_normalization/batchnorm/subSub2Encoder/second_layer/batch_normalization/beta/read8Encoder/second_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:*
T0
ф
8Encoder/second_layer/batch_normalization/batchnorm/add_1Add8Encoder/second_layer/batch_normalization/batchnorm/mul_16Encoder/second_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:џџџџџџџџџ
j
%Encoder/second_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
О
#Encoder/second_layer/leaky_relu/mulMul%Encoder/second_layer/leaky_relu/alpha8Encoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
М
Encoder/second_layer/leaky_reluMaximum#Encoder/second_layer/leaky_relu/mul8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
Й
:Encoder/encoder_mu/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
Ћ
8Encoder/encoder_mu/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *?ШЪН*
dtype0*
_output_shapes
: 
Ћ
8Encoder/encoder_mu/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *?ШЪ=*
dtype0*
_output_shapes
: 

BEncoder/encoder_mu/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Encoder/encoder_mu/kernel/Initializer/random_uniform/shape*

seed *
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
seed2 *
dtype0*
_output_shapes
:	d

8Encoder/encoder_mu/kernel/Initializer/random_uniform/subSub8Encoder/encoder_mu/kernel/Initializer/random_uniform/max8Encoder/encoder_mu/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
: 

8Encoder/encoder_mu/kernel/Initializer/random_uniform/mulMulBEncoder/encoder_mu/kernel/Initializer/random_uniform/RandomUniform8Encoder/encoder_mu/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	d

4Encoder/encoder_mu/kernel/Initializer/random_uniformAdd8Encoder/encoder_mu/kernel/Initializer/random_uniform/mul8Encoder/encoder_mu/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	d
Н
Encoder/encoder_mu/kernel
VariableV2*,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name 
ќ
 Encoder/encoder_mu/kernel/AssignAssignEncoder/encoder_mu/kernel4Encoder/encoder_mu/kernel/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	d

Encoder/encoder_mu/kernel/readIdentityEncoder/encoder_mu/kernel*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	d
Ђ
)Encoder/encoder_mu/bias/Initializer/zerosConst**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0*
_output_shapes
:d
Џ
Encoder/encoder_mu/bias
VariableV2**
_class 
loc:@Encoder/encoder_mu/bias*
	container *
shape:d*
dtype0*
_output_shapes
:d*
shared_name 
ц
Encoder/encoder_mu/bias/AssignAssignEncoder/encoder_mu/bias)Encoder/encoder_mu/bias/Initializer/zeros**
_class 
loc:@Encoder/encoder_mu/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0

Encoder/encoder_mu/bias/readIdentityEncoder/encoder_mu/bias*
_output_shapes
:d*
T0**
_class 
loc:@Encoder/encoder_mu/bias
М
Encoder/encoder_mu/MatMulMatMulEncoder/second_layer/leaky_reluEncoder/encoder_mu/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b( 
Ї
Encoder/encoder_mu/BiasAddBiasAddEncoder/encoder_mu/MatMulEncoder/encoder_mu/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџd
С
>Encoder/encoder_logvar/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0
Г
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/minConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *?ШЪН*
dtype0*
_output_shapes
: 
Г
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *?ШЪ=*
dtype0

FEncoder/encoder_logvar/kernel/Initializer/random_uniform/RandomUniformRandomUniform>Encoder/encoder_logvar/kernel/Initializer/random_uniform/shape*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
seed2 *
dtype0*
_output_shapes
:	d*

seed 

<Encoder/encoder_logvar/kernel/Initializer/random_uniform/subSub<Encoder/encoder_logvar/kernel/Initializer/random_uniform/max<Encoder/encoder_logvar/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel
Ѕ
<Encoder/encoder_logvar/kernel/Initializer/random_uniform/mulMulFEncoder/encoder_logvar/kernel/Initializer/random_uniform/RandomUniform<Encoder/encoder_logvar/kernel/Initializer/random_uniform/sub*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	d*
T0

8Encoder/encoder_logvar/kernel/Initializer/random_uniformAdd<Encoder/encoder_logvar/kernel/Initializer/random_uniform/mul<Encoder/encoder_logvar/kernel/Initializer/random_uniform/min*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	d*
T0
Х
Encoder/encoder_logvar/kernel
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	d

$Encoder/encoder_logvar/kernel/AssignAssignEncoder/encoder_logvar/kernel8Encoder/encoder_logvar/kernel/Initializer/random_uniform*
use_locking(*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(*
_output_shapes
:	d
Љ
"Encoder/encoder_logvar/kernel/readIdentityEncoder/encoder_logvar/kernel*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	d
Њ
-Encoder/encoder_logvar/bias/Initializer/zerosConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0*
_output_shapes
:d
З
Encoder/encoder_logvar/bias
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape:d
і
"Encoder/encoder_logvar/bias/AssignAssignEncoder/encoder_logvar/bias-Encoder/encoder_logvar/bias/Initializer/zeros*
_output_shapes
:d*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(

 Encoder/encoder_logvar/bias/readIdentityEncoder/encoder_logvar/bias*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
:d*
T0
Ф
Encoder/encoder_logvar/MatMulMatMulEncoder/second_layer/leaky_relu"Encoder/encoder_logvar/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b( 
Г
Encoder/encoder_logvar/BiasAddBiasAddEncoder/encoder_logvar/MatMul Encoder/encoder_logvar/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџd
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
 *  ?*
dtype0
Њ
*Encoder/random_normal/RandomStandardNormalRandomStandardNormalEncoder/random_normal/shape*
T0	*
dtype0*
_output_shapes
:d*
seed2 *

seed 

Encoder/random_normal/mulMul*Encoder/random_normal/RandomStandardNormalEncoder/random_normal/stddev*
T0*
_output_shapes
:d
x
Encoder/random_normalAddEncoder/random_normal/mulEncoder/random_normal/mean*
_output_shapes
:d*
T0
V
Encoder/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 

Encoder/truedivRealDivEncoder/encoder_logvar/BiasAddEncoder/truediv/y*'
_output_shapes
:џџџџџџџџџd*
T0
U
Encoder/ExpExpEncoder/truediv*'
_output_shapes
:џџџџџџџџџd*
T0
o
Encoder/logvar_stdMulEncoder/random_normalEncoder/Exp*'
_output_shapes
:џџџџџџџџџd*
T0
t
Encoder/AddAddEncoder/logvar_stdEncoder/encoder_mu/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџd
^
Encoder/encoder_codeSigmoidEncoder/Add*'
_output_shapes
:џџџџџџџџџd*
T0
л
KDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
Э
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *?ШЪН*
dtype0*
_output_shapes
: 
Э
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
valueB
 *?ШЪ=*
dtype0*
_output_shapes
: 
Р
SDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformKDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
seed2 *
dtype0*
_output_shapes
:	d*

seed 
Ц
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/maxIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
: 
й
IDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulSDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d*
T0
Ы
EDecoder/first_layer/fully_connected/kernel/Initializer/random_uniformAddIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/mulIDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d
п
*Decoder/first_layer/fully_connected/kernel
VariableV2*
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name *=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
	container 
Р
1Decoder/first_layer/fully_connected/kernel/AssignAssign*Decoder/first_layer/fully_connected/kernelEDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d
а
/Decoder/first_layer/fully_connected/kernel/readIdentity*Decoder/first_layer/fully_connected/kernel*
T0*=
_class3
1/loc:@Decoder/first_layer/fully_connected/kernel*
_output_shapes
:	d
Ц
:Decoder/first_layer/fully_connected/bias/Initializer/zerosConst*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
г
(Decoder/first_layer/fully_connected/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias
Ћ
/Decoder/first_layer/fully_connected/bias/AssignAssign(Decoder/first_layer/fully_connected/bias:Decoder/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ц
-Decoder/first_layer/fully_connected/bias/readIdentity(Decoder/first_layer/fully_connected/bias*
_output_shapes	
:*
T0*;
_class1
/-loc:@Decoder/first_layer/fully_connected/bias
д
*Decoder/first_layer/fully_connected/MatMulMatMulEncoder/encoder_code/Decoder/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
л
+Decoder/first_layer/fully_connected/BiasAddBiasAdd*Decoder/first_layer/fully_connected/MatMul-Decoder/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
i
$Decoder/first_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Џ
"Decoder/first_layer/leaky_relu/mulMul$Decoder/first_layer/leaky_relu/alpha+Decoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
­
Decoder/first_layer/leaky_reluMaximum"Decoder/first_layer/leaky_relu/mul+Decoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
н
LDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Я
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *qФН*
dtype0*
_output_shapes
: 
Я
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
valueB
 *qФ=*
dtype0*
_output_shapes
: 
Ф
TDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
seed2 
Ъ
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/maxJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
_output_shapes
: *
T0
о
JDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:

а
FDecoder/second_layer/fully_connected/kernel/Initializer/random_uniformAddJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/mulJDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:

у
+Decoder/second_layer/fully_connected/kernel
VariableV2* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0
Х
2Decoder/second_layer/fully_connected/kernel/AssignAssign+Decoder/second_layer/fully_connected/kernelFDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel*
validate_shape(
д
0Decoder/second_layer/fully_connected/kernel/readIdentity+Decoder/second_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Decoder/second_layer/fully_connected/kernel* 
_output_shapes
:

Ш
;Decoder/second_layer/fully_connected/bias/Initializer/zerosConst*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
е
)Decoder/second_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
	container *
shape:
Џ
0Decoder/second_layer/fully_connected/bias/AssignAssign)Decoder/second_layer/fully_connected/bias;Decoder/second_layer/fully_connected/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias*
validate_shape(
Щ
.Decoder/second_layer/fully_connected/bias/readIdentity)Decoder/second_layer/fully_connected/bias*
_output_shapes	
:*
T0*<
_class2
0.loc:@Decoder/second_layer/fully_connected/bias
р
+Decoder/second_layer/fully_connected/MatMulMatMulDecoder/first_layer/leaky_relu0Decoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
о
,Decoder/second_layer/fully_connected/BiasAddBiasAdd+Decoder/second_layer/fully_connected/MatMul.Decoder/second_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
б
?Decoder/second_layer/batch_normalization/gamma/Initializer/onesConst*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
п
.Decoder/second_layer/batch_normalization/gamma
VariableV2*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Т
5Decoder/second_layer/batch_normalization/gamma/AssignAssign.Decoder/second_layer/batch_normalization/gamma?Decoder/second_layer/batch_normalization/gamma/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
validate_shape(
и
3Decoder/second_layer/batch_normalization/gamma/readIdentity.Decoder/second_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Decoder/second_layer/batch_normalization/gamma*
_output_shapes	
:
а
?Decoder/second_layer/batch_normalization/beta/Initializer/zerosConst*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
н
-Decoder/second_layer/batch_normalization/beta
VariableV2*
shared_name *@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
П
4Decoder/second_layer/batch_normalization/beta/AssignAssign-Decoder/second_layer/batch_normalization/beta?Decoder/second_layer/batch_normalization/beta/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
validate_shape(
е
2Decoder/second_layer/batch_normalization/beta/readIdentity-Decoder/second_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Decoder/second_layer/batch_normalization/beta*
_output_shapes	
:
о
FDecoder/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*
_output_shapes	
:*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
valueB*    *
dtype0
ы
4Decoder/second_layer/batch_normalization/moving_mean
VariableV2*
shared_name *G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
л
;Decoder/second_layer/batch_normalization/moving_mean/AssignAssign4Decoder/second_layer/batch_normalization/moving_meanFDecoder/second_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:
ъ
9Decoder/second_layer/batch_normalization/moving_mean/readIdentity4Decoder/second_layer/batch_normalization/moving_mean*
_output_shapes	
:*
T0*G
_class=
;9loc:@Decoder/second_layer/batch_normalization/moving_mean
х
IDecoder/second_layer/batch_normalization/moving_variance/Initializer/onesConst*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ѓ
8Decoder/second_layer/batch_normalization/moving_variance
VariableV2*
shared_name *K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
ъ
?Decoder/second_layer/batch_normalization/moving_variance/AssignAssign8Decoder/second_layer/batch_normalization/moving_varianceIDecoder/second_layer/batch_normalization/moving_variance/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
validate_shape(
і
=Decoder/second_layer/batch_normalization/moving_variance/readIdentity8Decoder/second_layer/batch_normalization/moving_variance*K
_classA
?=loc:@Decoder/second_layer/batch_normalization/moving_variance*
_output_shapes	
:*
T0
}
8Decoder/second_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
м
6Decoder/second_layer/batch_normalization/batchnorm/addAdd=Decoder/second_layer/batch_normalization/moving_variance/read8Decoder/second_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:*
T0

8Decoder/second_layer/batch_normalization/batchnorm/RsqrtRsqrt6Decoder/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
в
6Decoder/second_layer/batch_normalization/batchnorm/mulMul8Decoder/second_layer/batch_normalization/batchnorm/Rsqrt3Decoder/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
и
8Decoder/second_layer/batch_normalization/batchnorm/mul_1Mul,Decoder/second_layer/fully_connected/BiasAdd6Decoder/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
и
8Decoder/second_layer/batch_normalization/batchnorm/mul_2Mul9Decoder/second_layer/batch_normalization/moving_mean/read6Decoder/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:*
T0
б
6Decoder/second_layer/batch_normalization/batchnorm/subSub2Decoder/second_layer/batch_normalization/beta/read8Decoder/second_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ф
8Decoder/second_layer/batch_normalization/batchnorm/add_1Add8Decoder/second_layer/batch_normalization/batchnorm/mul_16Decoder/second_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:џџџџџџџџџ*
T0
j
%Decoder/second_layer/leaky_relu/alphaConst*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
О
#Decoder/second_layer/leaky_relu/mulMul%Decoder/second_layer/leaky_relu/alpha8Decoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
М
Decoder/second_layer/leaky_reluMaximum#Decoder/second_layer/leaky_relu/mul8Decoder/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
Џ
5Decoder/dense/kernel/Initializer/random_uniform/shapeConst*'
_class
loc:@Decoder/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ё
3Decoder/dense/kernel/Initializer/random_uniform/minConst*'
_class
loc:@Decoder/dense/kernel*
valueB
 *HYН*
dtype0*
_output_shapes
: 
Ё
3Decoder/dense/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@Decoder/dense/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
џ
=Decoder/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5Decoder/dense/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*'
_class
loc:@Decoder/dense/kernel*
seed2 
ю
3Decoder/dense/kernel/Initializer/random_uniform/subSub3Decoder/dense/kernel/Initializer/random_uniform/max3Decoder/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@Decoder/dense/kernel

3Decoder/dense/kernel/Initializer/random_uniform/mulMul=Decoder/dense/kernel/Initializer/random_uniform/RandomUniform3Decoder/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:

є
/Decoder/dense/kernel/Initializer/random_uniformAdd3Decoder/dense/kernel/Initializer/random_uniform/mul3Decoder/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:

Е
Decoder/dense/kernel
VariableV2*'
_class
loc:@Decoder/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
щ
Decoder/dense/kernel/AssignAssignDecoder/dense/kernel/Decoder/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@Decoder/dense/kernel*
validate_shape(* 
_output_shapes
:


Decoder/dense/kernel/readIdentityDecoder/dense/kernel*
T0*'
_class
loc:@Decoder/dense/kernel* 
_output_shapes
:


$Decoder/dense/bias/Initializer/zerosConst*%
_class
loc:@Decoder/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ї
Decoder/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *%
_class
loc:@Decoder/dense/bias*
	container 
г
Decoder/dense/bias/AssignAssignDecoder/dense/bias$Decoder/dense/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*%
_class
loc:@Decoder/dense/bias*
validate_shape(

Decoder/dense/bias/readIdentityDecoder/dense/bias*%
_class
loc:@Decoder/dense/bias*
_output_shapes	
:*
T0
Г
Decoder/dense/MatMulMatMulDecoder/second_layer/leaky_reluDecoder/dense/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

Decoder/dense/BiasAddBiasAddDecoder/dense/MatMulDecoder/dense/bias/read*(
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
d
Decoder/last_layerTanhDecoder/dense/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
t
Decoder/reshape_image/shapeConst*
_output_shapes
:*%
valueB"џџџџ         *
dtype0

Decoder/reshape_imageReshapeDecoder/last_layerDecoder/reshape_image/shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
~
Discriminator/noise_code_inPlaceholder*'
_output_shapes
:џџџџџџџџџd*
shape:џџџџџџџџџd*
dtype0
ч
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
й
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *?ШЪН*
dtype0*
_output_shapes
: 
й
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *?ШЪ=*
dtype0*
_output_shapes
: 
в
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d*

seed *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
seed2 
о
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
ё
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d
у
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d
ы
0Discriminator/first_layer/fully_connected/kernel
VariableV2*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
и
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d
т
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
в
@Discriminator/first_layer/fully_connected/bias/Initializer/zerosConst*
_output_shapes	
:*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0
п
.Discriminator/first_layer/fully_connected/bias
VariableV2*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
У
5Discriminator/first_layer/fully_connected/bias/AssignAssign.Discriminator/first_layer/fully_connected/bias@Discriminator/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
и
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:*
T0
р
0Discriminator/first_layer/fully_connected/MatMulMatMulEncoder/encoder_code5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
э
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*(
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
o
*Discriminator/first_layer/leaky_relu/alphaConst*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
С
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
П
$Discriminator/first_layer/leaky_reluMaximum(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
щ
RDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
л
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *ѓЕН*
dtype0*
_output_shapes
: 
л
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *ѓЕ=*
dtype0*
_output_shapes
: 
ж
ZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformRDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*

seed *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
seed2 *
dtype0
т
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
і
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
ш
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

я
1Discriminator/second_layer/fully_connected/kernel
VariableV2*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

н
8Discriminator/second_layer/fully_connected/kernel/AssignAssign1Discriminator/second_layer/fully_connected/kernelLDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

ц
6Discriminator/second_layer/fully_connected/kernel/readIdentity1Discriminator/second_layer/fully_connected/kernel*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
д
ADiscriminator/second_layer/fully_connected/bias/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
с
/Discriminator/second_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:
Ч
6Discriminator/second_layer/fully_connected/bias/AssignAssign/Discriminator/second_layer/fully_connected/biasADiscriminator/second_layer/fully_connected/bias/Initializer/zeros*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
л
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
ђ
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
№
2Discriminator/second_layer/fully_connected/BiasAddBiasAdd1Discriminator/second_layer/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
p
+Discriminator/second_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ф
)Discriminator/second_layer/leaky_relu/mulMul+Discriminator/second_layer/leaky_relu/alpha2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
Т
%Discriminator/second_layer/leaky_reluMaximum)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Й
:Discriminator/prob/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ћ
8Discriminator/prob/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *IvО*
dtype0*
_output_shapes
: 
Ћ
8Discriminator/prob/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 

BDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Discriminator/prob/kernel/Initializer/random_uniform/shape*

seed *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
seed2 *
dtype0*
_output_shapes
:	

8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
: *
T0

8Discriminator/prob/kernel/Initializer/random_uniform/mulMulBDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniform8Discriminator/prob/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	

4Discriminator/prob/kernel/Initializer/random_uniformAdd8Discriminator/prob/kernel/Initializer/random_uniform/mul8Discriminator/prob/kernel/Initializer/random_uniform/min*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	*
T0
Н
Discriminator/prob/kernel
VariableV2*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
ќ
 Discriminator/prob/kernel/AssignAssignDiscriminator/prob/kernel4Discriminator/prob/kernel/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	

Discriminator/prob/kernel/readIdentityDiscriminator/prob/kernel*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	
Ђ
)Discriminator/prob/bias/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
Џ
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
ц
Discriminator/prob/bias/AssignAssignDiscriminator/prob/bias)Discriminator/prob/bias/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:

Discriminator/prob/bias/readIdentityDiscriminator/prob/bias*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
Т
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ї
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
щ
2Discriminator/first_layer_1/fully_connected/MatMulMatMulDiscriminator/noise_code_in5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
ё
3Discriminator/first_layer_1/fully_connected/BiasAddBiasAdd2Discriminator/first_layer_1/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
q
,Discriminator/first_layer_1/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ч
*Discriminator/first_layer_1/leaky_relu/mulMul,Discriminator/first_layer_1/leaky_relu/alpha3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
Х
&Discriminator/first_layer_1/leaky_reluMaximum*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
і
3Discriminator/second_layer_1/fully_connected/MatMulMatMul&Discriminator/first_layer_1/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
є
4Discriminator/second_layer_1/fully_connected/BiasAddBiasAdd3Discriminator/second_layer_1/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*(
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
r
-Discriminator/second_layer_1/leaky_relu/alphaConst*
_output_shapes
: *
valueB
 *ЭЬL>*
dtype0
Ъ
+Discriminator/second_layer_1/leaky_relu/mulMul-Discriminator/second_layer_1/leaky_relu/alpha4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Ш
'Discriminator/second_layer_1/leaky_reluMaximum+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Ц
Discriminator/prob_1/MatMulMatMul'Discriminator/second_layer_1/leaky_reluDiscriminator/prob/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ћ
Discriminator/prob_1/BiasAddBiasAddDiscriminator/prob_1/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
i
ones_like/ShapeShapeDiscriminator/prob/BiasAdd*
_output_shapes
:*
T0*
out_type0
T
ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
w
	ones_likeFillones_like/Shapeones_like/Const*
T0*

index_type0*'
_output_shapes
:џџџџџџџџџ
s
logistic_loss/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0

logistic_loss/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
Ђ
logistic_loss/SelectSelectlogistic_loss/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
f
logistic_loss/NegNegDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0

logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/NegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
q
logistic_loss/mulMulDiscriminator/prob/BiasAdd	ones_like*
T0*'
_output_shapes
:џџџџџџџџџ
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0*'
_output_shapes
:џџџџџџџџџ
b
logistic_loss/ExpExplogistic_loss/Select_1*
T0*'
_output_shapes
:џџџџџџџџџ
a
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*'
_output_shapes
:џџџџџџџџџ
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0*'
_output_shapes
:џџџџџџџџџ
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
m
adversalrial_lossMeanlogistic_lossConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
l
SubSubDecoder/reshape_imageEncoder/real_in*
T0*/
_output_shapes
:џџџџџџџџџ
I
AbsAbsSub*
T0*/
_output_shapes
:џџџџџџџџџ
`
Const_1Const*%
valueB"             *
dtype0*
_output_shapes
:
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
 *o:*
dtype0*
_output_shapes
: 
E
mulMulmul/xadversalrial_loss*
_output_shapes
: *
T0
L
mul_1/xConst*
valueB
 *wО?*
dtype0*
_output_shapes
: 
F
mul_1Mulmul_1/xpixelwise_loss*
_output_shapes
: *
T0
B
generator_lossAddmulmul_1*
T0*
_output_shapes
: 
e
generator_loss_1/tagConst*
_output_shapes
: *!
valueB Bgenerator_loss_1*
dtype0
k
generator_loss_1HistogramSummarygenerator_loss_1/taggenerator_loss*
_output_shapes
: *
T0
m
ones_like_1/ShapeShapeDiscriminator/prob_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
V
ones_like_1/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
}
ones_like_1Fillones_like_1/Shapeones_like_1/Const*'
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
w
logistic_loss_1/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

logistic_loss_1/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
j
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Ѕ
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
w
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAddones_like_1*'
_output_shapes
:џџџџџџџџџ*
T0
y
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*'
_output_shapes
:џџџџџџџџџ*
T0
f
logistic_loss_1/ExpExplogistic_loss_1/Select_1*'
_output_shapes
:џџџџџџџџџ*
T0
e
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*'
_output_shapes
:џџџџџџџџџ*
T0
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*
T0*'
_output_shapes
:џџџџџџџџџ
X
Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
d
MeanMeanlogistic_loss_1Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e

zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
u
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0

logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
Ј
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
h
logistic_loss_2/NegNegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
Ѓ
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
t
logistic_loss_2/mulMulDiscriminator/prob/BiasAdd
zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
y
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*
T0*'
_output_shapes
:џџџџџџџџџ
f
logistic_loss_2/ExpExplogistic_loss_2/Select_1*'
_output_shapes
:џџџџџџџџџ*
T0
e
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*
T0*'
_output_shapes
:џџџџџџџџџ
t
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*'
_output_shapes
:џџџџџџџџџ*
T0
X
Const_3Const*
valueB"       *
dtype0*
_output_shapes
:
f
Mean_1Meanlogistic_loss_2Const_3*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
9
addAddMeanMean_1*
_output_shapes
: *
T0
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
 *  ?*
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
Б
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Fill$^gradients/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
Г
-gradients/add_grad/tuple/control_dependency_1Identitygradients/Fill$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
­
gradients/Mean_grad/ReshapeReshape+gradients/add_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
h
gradients/Mean_grad/ShapeShapelogistic_loss_1*
out_type0*
_output_shapes
:*
T0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

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

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

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

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
t
#gradients/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
Г
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
Ђ
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
l
gradients/Mean_1_grad/Shape_1Shapelogistic_loss_2*
out_type0*
_output_shapes
:*
T0
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

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
 
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

gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
_output_shapes
: *
T0

gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ
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
и
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
М
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Р
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

/gradients/logistic_loss_1_grad/tuple/group_depsNoOp'^gradients/logistic_loss_1_grad/Reshape)^gradients/logistic_loss_1_grad/Reshape_1

7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0

9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
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
и
4gradients/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_2_grad/Shape&gradients/logistic_loss_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
О
"gradients/logistic_loss_2_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Л
&gradients/logistic_loss_2_grad/ReshapeReshape"gradients/logistic_loss_2_grad/Sum$gradients/logistic_loss_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Т
$gradients/logistic_loss_2_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(gradients/logistic_loss_2_grad/Reshape_1Reshape$gradients/logistic_loss_2_grad/Sum_1&gradients/logistic_loss_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

/gradients/logistic_loss_2_grad/tuple/group_depsNoOp'^gradients/logistic_loss_2_grad/Reshape)^gradients/logistic_loss_2_grad/Reshape_1

7gradients/logistic_loss_2_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_2_grad/Reshape0^gradients/logistic_loss_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*9
_class/
-+loc:@gradients/logistic_loss_2_grad/Reshape

9gradients/logistic_loss_2_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_2_grad/Reshape_10^gradients/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
~
(gradients/logistic_loss_1/sub_grad/ShapeShapelogistic_loss_1/Select*
T0*
out_type0*
_output_shapes
:
}
*gradients/logistic_loss_1/sub_grad/Shape_1Shapelogistic_loss_1/mul*
out_type0*
_output_shapes
:*
T0
ф
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
р
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ч
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ф
(gradients/logistic_loss_1/sub_grad/Sum_1Sum7gradients/logistic_loss_1_grad/tuple/control_dependency:gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
z
&gradients/logistic_loss_1/sub_grad/NegNeg(gradients/logistic_loss_1/sub_grad/Sum_1*
_output_shapes
:*
T0
Ы
,gradients/logistic_loss_1/sub_grad/Reshape_1Reshape&gradients/logistic_loss_1/sub_grad/Neg*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3gradients/logistic_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/sub_grad/Reshape-^gradients/logistic_loss_1/sub_grad/Reshape_1

;gradients/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/sub_grad/Reshape4^gradients/logistic_loss_1/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/sub_grad/Reshape
 
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ћ
*gradients/logistic_loss_1/Log1p_grad/add/xConst:^gradients/logistic_loss_1_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ђ
(gradients/logistic_loss_1/Log1p_grad/addAdd*gradients/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*'
_output_shapes
:џџџџџџџџџ*
T0
Э
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ
~
(gradients/logistic_loss_2/sub_grad/ShapeShapelogistic_loss_2/Select*
T0*
out_type0*
_output_shapes
:
}
*gradients/logistic_loss_2/sub_grad/Shape_1Shapelogistic_loss_2/mul*
out_type0*
_output_shapes
:*
T0
ф
8gradients/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_2/sub_grad/Shape*gradients/logistic_loss_2/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
р
&gradients/logistic_loss_2/sub_grad/SumSum7gradients/logistic_loss_2_grad/tuple/control_dependency8gradients/logistic_loss_2/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ч
*gradients/logistic_loss_2/sub_grad/ReshapeReshape&gradients/logistic_loss_2/sub_grad/Sum(gradients/logistic_loss_2/sub_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ф
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
Ы
,gradients/logistic_loss_2/sub_grad/Reshape_1Reshape&gradients/logistic_loss_2/sub_grad/Neg*gradients/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3gradients/logistic_loss_2/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_2/sub_grad/Reshape-^gradients/logistic_loss_2/sub_grad/Reshape_1

;gradients/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_2/sub_grad/Reshape4^gradients/logistic_loss_2/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_2/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
 
=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_2/sub_grad/Reshape_14^gradients/logistic_loss_2/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/logistic_loss_2/sub_grad/Reshape_1
Ћ
*gradients/logistic_loss_2/Log1p_grad/add/xConst:^gradients/logistic_loss_2_grad/tuple/control_dependency_1*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ђ
(gradients/logistic_loss_2/Log1p_grad/addAdd*gradients/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_2/Log1p_grad/add*'
_output_shapes
:џџџџџџџџџ*
T0
Э
(gradients/logistic_loss_2/Log1p_grad/mulMul9gradients/logistic_loss_2_grad/tuple/control_dependency_1/gradients/logistic_loss_2/Log1p_grad/Reciprocal*'
_output_shapes
:џџџџџџџџџ*
T0

0gradients/logistic_loss_1/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
ѕ
,gradients/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual;gradients/logistic_loss_1/sub_grad/tuple/control_dependency0gradients/logistic_loss_1/Select_grad/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
ї
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
Є
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ*
T0
Њ
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_grad/Select_1

(gradients/logistic_loss_1/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
u
*gradients/logistic_loss_1/mul_grad/Shape_1Shapeones_like_1*
T0*
out_type0*
_output_shapes
:
ф
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1ones_like_1*'
_output_shapes
:џџџџџџџџџ*
T0
Я
&gradients/logistic_loss_1/mul_grad/SumSum&gradients/logistic_loss_1/mul_grad/Mul8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ч
*gradients/logistic_loss_1/mul_grad/ReshapeReshape&gradients/logistic_loss_1/mul_grad/Sum(gradients/logistic_loss_1/mul_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
О
(gradients/logistic_loss_1/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1*'
_output_shapes
:џџџџџџџџџ*
T0
е
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Э
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

3gradients/logistic_loss_1/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/mul_grad/Reshape-^gradients/logistic_loss_1/mul_grad/Reshape_1

;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/mul_grad/Reshape4^gradients/logistic_loss_1/mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/mul_grad/Reshape
 
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1

&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

0gradients/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
ѕ
,gradients/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual;gradients/logistic_loss_2/sub_grad/tuple/control_dependency0gradients/logistic_loss_2/Select_grad/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
ї
.gradients/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual0gradients/logistic_loss_2/Select_grad/zeros_like;gradients/logistic_loss_2/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

6gradients/logistic_loss_2/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_2/Select_grad/Select/^gradients/logistic_loss_2/Select_grad/Select_1
Є
>gradients/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_2/Select_grad/Select7^gradients/logistic_loss_2/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Њ
@gradients/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_2/Select_grad/Select_17^gradients/logistic_loss_2/Select_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@gradients/logistic_loss_2/Select_grad/Select_1

(gradients/logistic_loss_2/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
_output_shapes
:*
T0*
out_type0
t
*gradients/logistic_loss_2/mul_grad/Shape_1Shape
zeros_like*
out_type0*
_output_shapes
:*
T0
ф
8gradients/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_2/mul_grad/Shape*gradients/logistic_loss_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Њ
&gradients/logistic_loss_2/mul_grad/MulMul=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1
zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Я
&gradients/logistic_loss_2/mul_grad/SumSum&gradients/logistic_loss_2/mul_grad/Mul8gradients/logistic_loss_2/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ч
*gradients/logistic_loss_2/mul_grad/ReshapeReshape&gradients/logistic_loss_2/mul_grad/Sum(gradients/logistic_loss_2/mul_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
М
(gradients/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd=gradients/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
е
(gradients/logistic_loss_2/mul_grad/Sum_1Sum(gradients/logistic_loss_2/mul_grad/Mul_1:gradients/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Э
,gradients/logistic_loss_2/mul_grad/Reshape_1Reshape(gradients/logistic_loss_2/mul_grad/Sum_1*gradients/logistic_loss_2/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

3gradients/logistic_loss_2/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_2/mul_grad/Reshape-^gradients/logistic_loss_2/mul_grad/Reshape_1

;gradients/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_2/mul_grad/Reshape4^gradients/logistic_loss_2/mul_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss_2/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
 
=gradients/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_2/mul_grad/Reshape_14^gradients/logistic_loss_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

&gradients/logistic_loss_2/Exp_grad/mulMul(gradients/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*'
_output_shapes
:џџџџџџџџџ*
T0

2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*
T0*'
_output_shapes
:џџџџџџџџџ
ф
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
ц
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Є
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
Ќ
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select*'
_output_shapes
:џџџџџџџџџ
В
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1

2gradients/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*'
_output_shapes
:џџџџџџџџџ*
T0
ф
.gradients/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual&gradients/logistic_loss_2/Exp_grad/mul2gradients/logistic_loss_2/Select_1_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
ц
0gradients/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients/logistic_loss_2/Select_1_grad/zeros_like&gradients/logistic_loss_2/Exp_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Є
8gradients/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_2/Select_1_grad/Select1^gradients/logistic_loss_2/Select_1_grad/Select_1
Ќ
@gradients/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_2/Select_1_grad/Select9^gradients/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_2/Select_1_grad/Select*'
_output_shapes
:џџџџџџџџџ
В
Bgradients/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_2/Select_1_grad/Select_19^gradients/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_2/Select_1_grad/Select_1*'
_output_shapes
:џџџџџџџџџ
Ё
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
&gradients/logistic_loss_2/Neg_grad/NegNeg@gradients/logistic_loss_2/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ћ
gradients/AddNAddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N

7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
_output_shapes
:*
T0

<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN8^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad

Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ*
T0
Л
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
§
gradients/AddN_1AddN>gradients/logistic_loss_2/Select_grad/tuple/control_dependency;gradients/logistic_loss_2/mul_grad/tuple/control_dependencyBgradients/logistic_loss_2/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_2/Neg_grad/Neg*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select*
N*'
_output_shapes
:џџџџџџџџџ*
T0

5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
T0*
data_formatNHWC*
_output_shapes
:

:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_16^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad

Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/logistic_loss_2/Select_grad/Select
Г
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
њ
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
ќ
3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
­
;gradients/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp2^gradients/Discriminator/prob_1/MatMul_grad/MatMul4^gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
Й
Cgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/Discriminator/prob_1/MatMul_grad/MatMul<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*D
_class:
86loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ*
T0
Ж
Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
і
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
і
1gradients/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
Ї
9gradients/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp0^gradients/Discriminator/prob/MatMul_grad/MatMul2^gradients/Discriminator/prob/MatMul_grad/MatMul_1
Б
Agradients/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity/gradients/Discriminator/prob/MatMul_grad/MatMul:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*B
_class8
64loc:@gradients/Discriminator/prob/MatMul_grad/MatMul
Ў
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	
 
gradients/AddN_2AddNFgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:*
T0
Ї
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
В
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
С
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosFill>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:џџџџџџџџџ*
T0
щ
Cgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
 
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Т
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
Ф
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape:gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
г
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
ы
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
Ѓ
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
Ў
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Н
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

:gradients/Discriminator/second_layer/leaky_relu_grad/zerosFill<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
у
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

Jgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Discriminator/second_layer/leaky_relu_grad/Shape<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
К
;gradients/Discriminator/second_layer/leaky_relu_grad/SelectSelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency:gradients/Discriminator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
М
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

8gradients/Discriminator/second_layer/leaky_relu_grad/SumSum;gradients/Discriminator/second_layer/leaky_relu_grad/SelectJgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ў
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1Lgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Э
Egradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp=^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape?^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
у
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ*
T0
щ
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1

gradients/AddN_3AddNEgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1*
N*
_output_shapes
:	

@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ж
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ќ
Pgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
џ
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ў
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
њ
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Rgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Dgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1Reshape@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
п
Kgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeE^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
щ
Sgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeL^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 

Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1L^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*W
_classM
KIloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1

>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
В
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
І
Ngradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulMulMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ј
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
є
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Pgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Bgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
й
Igradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpA^gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeC^gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
с
Qgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeJ^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape
љ
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
г
gradients/AddN_4AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N
­
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:
С
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
й
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Э
gradients/AddN_5AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0
Ћ
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:
Н
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
г
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1

\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Т
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ќ
Kgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ѕ
Sgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulL^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1

[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulT^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
О
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
І
Igradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_reluZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
я
Qgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpH^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulJ^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1

Ygradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityGgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulR^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
щ
gradients/AddN_6AddN^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
N
Ѕ
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
А
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
и
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

;gradients/Discriminator/first_layer_1/leaky_relu_grad/zerosFill=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
ц
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
з
<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
й
>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients/Discriminator/first_layer_1/leaky_relu_grad/SumSum<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectKgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0

;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
а
Fgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp>^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape@^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
ч
Ngradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeG^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
э
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Ё
9gradients/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ќ
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
д
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2ShapeYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0

?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

9gradients/Discriminator/first_layer/leaky_relu_grad/zerosFill;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
р
@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
:gradients/Discriminator/first_layer/leaky_relu_grad/SelectSelect@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency9gradients/Discriminator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
б
<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Select@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqual9gradients/Discriminator/first_layer/leaky_relu_grad/zerosYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

7gradients/Discriminator/first_layer/leaky_relu_grad/SumSum:gradients/Discriminator/first_layer/leaky_relu_grad/SelectIgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ћ
;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape7gradients/Discriminator/first_layer/leaky_relu_grad/Sum9gradients/Discriminator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Kgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ъ
Dgradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp<^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape>^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
п
Lgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeE^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape
х
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
ш
gradients/AddN_7AddN]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:


?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Д
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
Љ
Ogradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ќ
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulOgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ћ
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ї
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Qgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Cgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1Reshape?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
м
Jgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeD^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
х
Rgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeK^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
§
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ*
T0

=gradients/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
А
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ѓ
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
і
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ѕ
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
ё
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Ogradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Agradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
ж
Hgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp@^gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeB^gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
н
Pgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeI^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
ѕ
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
а
gradients/AddN_8AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0
Ќ
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:
П
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ж
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ъ
gradients/AddN_9AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:џџџџџџџџџ*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N
Њ
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:
Л
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
а
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradR^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
О
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(*
T0

Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/noise_code_in[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d*
transpose_a(*
transpose_b( *
T0
ђ
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1

Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџd

\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1S^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
_output_shapes
:	d*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
К
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(*
T0

Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_codeYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d*
transpose_a(*
transpose_b( *
T0
ь
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1

Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџd*
T0

Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d
ч
gradients/AddN_10AddN]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:*
T0
х
gradients/AddN_11AddN\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1*
N*
_output_shapes
:	d
Ё
beta1_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *   ?*
dtype0*
_output_shapes
: 
В
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
б
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 

beta1_power/readIdentitybeta1_power*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
Ё
beta2_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *wО?*
dtype0*
_output_shapes
: 
В
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
б
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(

beta2_power/readIdentitybeta2_power*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
ћ
eDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
х
[Discriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/ConstConst*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
Ђ
UDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zerosFilleDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensor[Discriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/Const*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d*
T0
ў
CDiscriminator/first_layer/fully_connected/kernel/discriminator_opti
VariableV2*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel

JDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/AssignAssignCDiscriminator/first_layer/fully_connected/kernel/discriminator_optiUDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0

HDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/readIdentityCDiscriminator/first_layer/fully_connected/kernel/discriminator_opti*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d
§
gDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0
ч
]Discriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/ConstConst*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
Ј
WDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zerosFillgDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensor]Discriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/Const*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d*
T0

EDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1
VariableV2*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d

LDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/AssignAssignEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1WDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0

JDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/readIdentityEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
:	d
х
SDiscriminator/first_layer/fully_connected/bias/discriminator_opti/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ђ
ADiscriminator/first_layer/fully_connected/bias/discriminator_opti
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
ќ
HDiscriminator/first_layer/fully_connected/bias/discriminator_opti/AssignAssignADiscriminator/first_layer/fully_connected/bias/discriminator_optiSDiscriminator/first_layer/fully_connected/bias/discriminator_opti/Initializer/zeros*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ў
FDiscriminator/first_layer/fully_connected/bias/discriminator_opti/readIdentityADiscriminator/first_layer/fully_connected/bias/discriminator_opti*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:*
T0
ч
UDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
є
CDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:

JDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/AssignAssignCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1UDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:

HDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/readIdentityCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:*
T0
§
fDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ч
\Discriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ї
VDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zerosFillfDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/shape_as_tensor\Discriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:


DDiscriminator/second_layer/fully_connected/kernel/discriminator_opti
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel

KDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/AssignAssignDDiscriminator/second_layer/fully_connected/kernel/discriminator_optiVDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

IDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/readIdentityDDiscriminator/second_layer/fully_connected/kernel/discriminator_opti*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

џ
hDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0
щ
^Discriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
­
XDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zerosFillhDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/shape_as_tensor^Discriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros/Const*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
*
T0

FDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1
VariableV2* 
_output_shapes
:
*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0

MDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/AssignAssignFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1XDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(

KDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/readIdentityFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
ч
TDiscriminator/second_layer/fully_connected/bias/discriminator_opti/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
є
BDiscriminator/second_layer/fully_connected/bias/discriminator_opti
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:

IDiscriminator/second_layer/fully_connected/bias/discriminator_opti/AssignAssignBDiscriminator/second_layer/fully_connected/bias/discriminator_optiTDiscriminator/second_layer/fully_connected/bias/discriminator_opti/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:

GDiscriminator/second_layer/fully_connected/bias/discriminator_opti/readIdentityBDiscriminator/second_layer/fully_connected/bias/discriminator_opti*
_output_shapes	
:*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
щ
VDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
і
DDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container 

KDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/AssignAssignDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1VDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(

IDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/readIdentityDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
У
>Discriminator/prob/kernel/discriminator_opti/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
а
,Discriminator/prob/kernel/discriminator_opti
VariableV2*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ќ
3Discriminator/prob/kernel/discriminator_opti/AssignAssign,Discriminator/prob/kernel/discriminator_opti>Discriminator/prob/kernel/discriminator_opti/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	
У
1Discriminator/prob/kernel/discriminator_opti/readIdentity,Discriminator/prob/kernel/discriminator_opti*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	
Х
@Discriminator/prob/kernel/discriminator_opti_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
в
.Discriminator/prob/kernel/discriminator_opti_1
VariableV2*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
В
5Discriminator/prob/kernel/discriminator_opti_1/AssignAssign.Discriminator/prob/kernel/discriminator_opti_1@Discriminator/prob/kernel/discriminator_opti_1/Initializer/zeros*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
Ч
3Discriminator/prob/kernel/discriminator_opti_1/readIdentity.Discriminator/prob/kernel/discriminator_opti_1*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	
Е
<Discriminator/prob/bias/discriminator_opti/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
Т
*Discriminator/prob/bias/discriminator_opti
VariableV2*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0*
_output_shapes
:

1Discriminator/prob/bias/discriminator_opti/AssignAssign*Discriminator/prob/bias/discriminator_opti<Discriminator/prob/bias/discriminator_opti/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
И
/Discriminator/prob/bias/discriminator_opti/readIdentity*Discriminator/prob/bias/discriminator_opti*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
З
>Discriminator/prob/bias/discriminator_opti_1/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
Ф
,Discriminator/prob/bias/discriminator_opti_1
VariableV2*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0
Ѕ
3Discriminator/prob/bias/discriminator_opti_1/AssignAssign,Discriminator/prob/bias/discriminator_opti_1>Discriminator/prob/bias/discriminator_opti_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(
М
1Discriminator/prob/bias/discriminator_opti_1/readIdentity,Discriminator/prob/bias/discriminator_opti_1*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
e
 discriminator_opti/learning_rateConst*
_output_shapes
: *
valueB
 *ЗQ9*
dtype0
]
discriminator_opti/beta1Const*
_output_shapes
: *
valueB
 *   ?*
dtype0
]
discriminator_opti/beta2Const*
valueB
 *wО?*
dtype0*
_output_shapes
: 
_
discriminator_opti/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
о
Tdiscriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam0Discriminator/first_layer/fully_connected/kernelCDiscriminator/first_layer/fully_connected/kernel/discriminator_optiEDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_11*
_output_shapes
:	d*
use_locking( *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
use_nesterov( 
а
Rdiscriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam.Discriminator/first_layer/fully_connected/biasADiscriminator/first_layer/fully_connected/bias/discriminator_optiCDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_10*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
у
Udiscriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernelDDiscriminator/second_layer/fully_connected/kernel/discriminator_optiFDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_7*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:

д
Sdiscriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/biasBDiscriminator/second_layer/fully_connected/bias/discriminator_optiDDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_6*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
ъ
=discriminator_opti/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernel,Discriminator/prob/kernel/discriminator_opti.Discriminator/prob/kernel/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_3*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
use_nesterov( *
_output_shapes
:	
л
;discriminator_opti/update_Discriminator/prob/bias/ApplyAdam	ApplyAdamDiscriminator/prob/bias*Discriminator/prob/bias/discriminator_opti,Discriminator/prob/bias/discriminator_opti_1beta1_power/readbeta2_power/read discriminator_opti/learning_ratediscriminator_opti/beta1discriminator_opti/beta2discriminator_opti/epsilongradients/AddN_2*
use_locking( *
T0**
_class 
loc:@Discriminator/prob/bias*
use_nesterov( *
_output_shapes
:

discriminator_opti/mulMulbeta1_power/readdiscriminator_opti/beta1S^discriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamU^discriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam<^discriminator_opti/update_Discriminator/prob/bias/ApplyAdam>^discriminator_opti/update_Discriminator/prob/kernel/ApplyAdamT^discriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamV^discriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
е
discriminator_opti/AssignAssignbeta1_powerdiscriminator_opti/mul*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 

discriminator_opti/mul_1Mulbeta2_power/readdiscriminator_opti/beta2S^discriminator_opti/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamU^discriminator_opti/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam<^discriminator_opti/update_Discriminator/prob/bias/ApplyAdam>^discriminator_opti/update_Discriminator/prob/kernel/ApplyAdamT^discriminator_opti/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamV^discriminator_opti/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
й
discriminator_opti/Assign_1Assignbeta2_powerdiscriminator_opti/mul_1*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
Ќ
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
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
K
0gradients_1/generator_loss_grad/tuple/group_depsNoOp^gradients_1/Fill
Я
8gradients_1/generator_loss_grad/tuple/control_dependencyIdentitygradients_1/Fill1^gradients_1/generator_loss_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill*
_output_shapes
: 
б
:gradients_1/generator_loss_grad/tuple/control_dependency_1Identitygradients_1/Fill1^gradients_1/generator_loss_grad/tuple/group_deps*#
_class
loc:@gradients_1/Fill*
_output_shapes
: *
T0

gradients_1/mul_grad/MulMul8gradients_1/generator_loss_grad/tuple/control_dependencyadversalrial_loss*
_output_shapes
: *
T0

gradients_1/mul_grad/Mul_1Mul8gradients_1/generator_loss_grad/tuple/control_dependencymul/x*
T0*
_output_shapes
: 
e
%gradients_1/mul_grad/tuple/group_depsNoOp^gradients_1/mul_grad/Mul^gradients_1/mul_grad/Mul_1
Щ
-gradients_1/mul_grad/tuple/control_dependencyIdentitygradients_1/mul_grad/Mul&^gradients_1/mul_grad/tuple/group_deps*+
_class!
loc:@gradients_1/mul_grad/Mul*
_output_shapes
: *
T0
Я
/gradients_1/mul_grad/tuple/control_dependency_1Identitygradients_1/mul_grad/Mul_1&^gradients_1/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*-
_class#
!loc:@gradients_1/mul_grad/Mul_1

gradients_1/mul_1_grad/MulMul:gradients_1/generator_loss_grad/tuple/control_dependency_1pixelwise_loss*
_output_shapes
: *
T0

gradients_1/mul_1_grad/Mul_1Mul:gradients_1/generator_loss_grad/tuple/control_dependency_1mul_1/x*
T0*
_output_shapes
: 
k
'gradients_1/mul_1_grad/tuple/group_depsNoOp^gradients_1/mul_1_grad/Mul^gradients_1/mul_1_grad/Mul_1
б
/gradients_1/mul_1_grad/tuple/control_dependencyIdentitygradients_1/mul_1_grad/Mul(^gradients_1/mul_1_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_1_grad/Mul*
_output_shapes
: 
з
1gradients_1/mul_1_grad/tuple/control_dependency_1Identitygradients_1/mul_1_grad/Mul_1(^gradients_1/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_1_grad/Mul_1*
_output_shapes
: 

0gradients_1/adversalrial_loss_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Я
*gradients_1/adversalrial_loss_grad/ReshapeReshape/gradients_1/mul_grad/tuple/control_dependency_10gradients_1/adversalrial_loss_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
u
(gradients_1/adversalrial_loss_grad/ShapeShapelogistic_loss*
T0*
out_type0*
_output_shapes
:
Щ
'gradients_1/adversalrial_loss_grad/TileTile*gradients_1/adversalrial_loss_grad/Reshape(gradients_1/adversalrial_loss_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
w
*gradients_1/adversalrial_loss_grad/Shape_1Shapelogistic_loss*
out_type0*
_output_shapes
:*
T0
m
*gradients_1/adversalrial_loss_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
r
(gradients_1/adversalrial_loss_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
У
'gradients_1/adversalrial_loss_grad/ProdProd*gradients_1/adversalrial_loss_grad/Shape_1(gradients_1/adversalrial_loss_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
t
*gradients_1/adversalrial_loss_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ч
)gradients_1/adversalrial_loss_grad/Prod_1Prod*gradients_1/adversalrial_loss_grad/Shape_2*gradients_1/adversalrial_loss_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
n
,gradients_1/adversalrial_loss_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Џ
*gradients_1/adversalrial_loss_grad/MaximumMaximum)gradients_1/adversalrial_loss_grad/Prod_1,gradients_1/adversalrial_loss_grad/Maximum/y*
_output_shapes
: *
T0
­
+gradients_1/adversalrial_loss_grad/floordivFloorDiv'gradients_1/adversalrial_loss_grad/Prod*gradients_1/adversalrial_loss_grad/Maximum*
_output_shapes
: *
T0

'gradients_1/adversalrial_loss_grad/CastCast+gradients_1/adversalrial_loss_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
Й
*gradients_1/adversalrial_loss_grad/truedivRealDiv'gradients_1/adversalrial_loss_grad/Tile'gradients_1/adversalrial_loss_grad/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

-gradients_1/pixelwise_loss_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
г
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
Ш
$gradients_1/pixelwise_loss_grad/TileTile'gradients_1/pixelwise_loss_grad/Reshape%gradients_1/pixelwise_loss_grad/Shape*/
_output_shapes
:џџџџџџџџџ*

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
К
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
О
&gradients_1/pixelwise_loss_grad/Prod_1Prod'gradients_1/pixelwise_loss_grad/Shape_2'gradients_1/pixelwise_loss_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
k
)gradients_1/pixelwise_loss_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
І
'gradients_1/pixelwise_loss_grad/MaximumMaximum&gradients_1/pixelwise_loss_grad/Prod_1)gradients_1/pixelwise_loss_grad/Maximum/y*
_output_shapes
: *
T0
Є
(gradients_1/pixelwise_loss_grad/floordivFloorDiv$gradients_1/pixelwise_loss_grad/Prod'gradients_1/pixelwise_loss_grad/Maximum*
_output_shapes
: *
T0

$gradients_1/pixelwise_loss_grad/CastCast(gradients_1/pixelwise_loss_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
И
'gradients_1/pixelwise_loss_grad/truedivRealDiv$gradients_1/pixelwise_loss_grad/Tile$gradients_1/pixelwise_loss_grad/Cast*
T0*/
_output_shapes
:џџџџџџџџџ
u
$gradients_1/logistic_loss_grad/ShapeShapelogistic_loss/sub*
_output_shapes
:*
T0*
out_type0
y
&gradients_1/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
out_type0*
_output_shapes
:*
T0
и
4gradients_1/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_1/logistic_loss_grad/Shape&gradients_1/logistic_loss_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ы
"gradients_1/logistic_loss_grad/SumSum*gradients_1/adversalrial_loss_grad/truediv4gradients_1/logistic_loss_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Л
&gradients_1/logistic_loss_grad/ReshapeReshape"gradients_1/logistic_loss_grad/Sum$gradients_1/logistic_loss_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Я
$gradients_1/logistic_loss_grad/Sum_1Sum*gradients_1/adversalrial_loss_grad/truediv6gradients_1/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
С
(gradients_1/logistic_loss_grad/Reshape_1Reshape$gradients_1/logistic_loss_grad/Sum_1&gradients_1/logistic_loss_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

/gradients_1/logistic_loss_grad/tuple/group_depsNoOp'^gradients_1/logistic_loss_grad/Reshape)^gradients_1/logistic_loss_grad/Reshape_1

7gradients_1/logistic_loss_grad/tuple/control_dependencyIdentity&gradients_1/logistic_loss_grad/Reshape0^gradients_1/logistic_loss_grad/tuple/group_deps*9
_class/
-+loc:@gradients_1/logistic_loss_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0

9gradients_1/logistic_loss_grad/tuple/control_dependency_1Identity(gradients_1/logistic_loss_grad/Reshape_10^gradients_1/logistic_loss_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_grad/Reshape_1
`
gradients_1/Abs_grad/SignSignSub*/
_output_shapes
:џџџџџџџџџ*
T0

gradients_1/Abs_grad/mulMul'gradients_1/pixelwise_loss_grad/truedivgradients_1/Abs_grad/Sign*/
_output_shapes
:џџџџџџџџџ*
T0
|
(gradients_1/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
out_type0*
_output_shapes
:*
T0
{
*gradients_1/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
out_type0*
_output_shapes
:*
T0
ф
8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/sub_grad/Shape*gradients_1/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
р
&gradients_1/logistic_loss/sub_grad/SumSum7gradients_1/logistic_loss_grad/tuple/control_dependency8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ч
*gradients_1/logistic_loss/sub_grad/ReshapeReshape&gradients_1/logistic_loss/sub_grad/Sum(gradients_1/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ф
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
Ы
,gradients_1/logistic_loss/sub_grad/Reshape_1Reshape&gradients_1/logistic_loss/sub_grad/Neg*gradients_1/logistic_loss/sub_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

3gradients_1/logistic_loss/sub_grad/tuple/group_depsNoOp+^gradients_1/logistic_loss/sub_grad/Reshape-^gradients_1/logistic_loss/sub_grad/Reshape_1

;gradients_1/logistic_loss/sub_grad/tuple/control_dependencyIdentity*gradients_1/logistic_loss/sub_grad/Reshape4^gradients_1/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
 
=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1Identity,gradients_1/logistic_loss/sub_grad/Reshape_14^gradients_1/logistic_loss/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ћ
*gradients_1/logistic_loss/Log1p_grad/add/xConst:^gradients_1/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0*
_output_shapes
: 
 
(gradients_1/logistic_loss/Log1p_grad/addAdd*gradients_1/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients_1/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_1/logistic_loss/Log1p_grad/add*'
_output_shapes
:џџџџџџџџџ*
T0
Э
(gradients_1/logistic_loss/Log1p_grad/mulMul9gradients_1/logistic_loss_grad/tuple/control_dependency_1/gradients_1/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ
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
К
*gradients_1/Sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Sub_grad/Shapegradients_1/Sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѕ
gradients_1/Sub_grad/SumSumgradients_1/Abs_grad/mul*gradients_1/Sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
gradients_1/Sub_grad/ReshapeReshapegradients_1/Sub_grad/Sumgradients_1/Sub_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
Љ
gradients_1/Sub_grad/Sum_1Sumgradients_1/Abs_grad/mul,gradients_1/Sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
^
gradients_1/Sub_grad/NegNeggradients_1/Sub_grad/Sum_1*
T0*
_output_shapes
:
Љ
gradients_1/Sub_grad/Reshape_1Reshapegradients_1/Sub_grad/Neggradients_1/Sub_grad/Shape_1*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ
m
%gradients_1/Sub_grad/tuple/group_depsNoOp^gradients_1/Sub_grad/Reshape^gradients_1/Sub_grad/Reshape_1
ъ
-gradients_1/Sub_grad/tuple/control_dependencyIdentitygradients_1/Sub_grad/Reshape&^gradients_1/Sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/Sub_grad/Reshape*/
_output_shapes
:џџџџџџџџџ
№
/gradients_1/Sub_grad/tuple/control_dependency_1Identitygradients_1/Sub_grad/Reshape_1&^gradients_1/Sub_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/Sub_grad/Reshape_1*/
_output_shapes
:џџџџџџџџџ*
T0

0gradients_1/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
,gradients_1/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual;gradients_1/logistic_loss/sub_grad/tuple/control_dependency0gradients_1/logistic_loss/Select_grad/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
ѕ
.gradients_1/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_1/logistic_loss/Select_grad/zeros_like;gradients_1/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

6gradients_1/logistic_loss/Select_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss/Select_grad/Select/^gradients_1/logistic_loss/Select_grad/Select_1
Є
>gradients_1/logistic_loss/Select_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss/Select_grad/Select7^gradients_1/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Њ
@gradients_1/logistic_loss/Select_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss/Select_grad/Select_17^gradients_1/logistic_loss/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss/Select_grad/Select_1*'
_output_shapes
:џџџџџџџџџ

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
ф
8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/mul_grad/Shape*gradients_1/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Љ
&gradients_1/logistic_loss/mul_grad/MulMul=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*'
_output_shapes
:џџџџџџџџџ*
T0
Я
&gradients_1/logistic_loss/mul_grad/SumSum&gradients_1/logistic_loss/mul_grad/Mul8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ч
*gradients_1/logistic_loss/mul_grad/ReshapeReshape&gradients_1/logistic_loss/mul_grad/Sum(gradients_1/logistic_loss/mul_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
М
(gradients_1/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1*'
_output_shapes
:џџџџџџџџџ*
T0
е
(gradients_1/logistic_loss/mul_grad/Sum_1Sum(gradients_1/logistic_loss/mul_grad/Mul_1:gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Э
,gradients_1/logistic_loss/mul_grad/Reshape_1Reshape(gradients_1/logistic_loss/mul_grad/Sum_1*gradients_1/logistic_loss/mul_grad/Shape_1*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

3gradients_1/logistic_loss/mul_grad/tuple/group_depsNoOp+^gradients_1/logistic_loss/mul_grad/Reshape-^gradients_1/logistic_loss/mul_grad/Reshape_1

;gradients_1/logistic_loss/mul_grad/tuple/control_dependencyIdentity*gradients_1/logistic_loss/mul_grad/Reshape4^gradients_1/logistic_loss/mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*=
_class3
1/loc:@gradients_1/logistic_loss/mul_grad/Reshape
 
=gradients_1/logistic_loss/mul_grad/tuple/control_dependency_1Identity,gradients_1/logistic_loss/mul_grad/Reshape_14^gradients_1/logistic_loss/mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients_1/logistic_loss/mul_grad/Reshape_1

&gradients_1/logistic_loss/Exp_grad/mulMul(gradients_1/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0*'
_output_shapes
:џџџџџџџџџ
~
,gradients_1/Decoder/reshape_image_grad/ShapeShapeDecoder/last_layer*
T0*
out_type0*
_output_shapes
:
з
.gradients_1/Decoder/reshape_image_grad/ReshapeReshape-gradients_1/Sub_grad/tuple/control_dependency,gradients_1/Decoder/reshape_image_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0

2gradients_1/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*'
_output_shapes
:џџџџџџџџџ*
T0
т
.gradients_1/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_1/logistic_loss/Exp_grad/mul2gradients_1/logistic_loss/Select_1_grad/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
ф
0gradients_1/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_1/logistic_loss/Select_1_grad/zeros_like&gradients_1/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Є
8gradients_1/logistic_loss/Select_1_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss/Select_1_grad/Select1^gradients_1/logistic_loss/Select_1_grad/Select_1
Ќ
@gradients_1/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss/Select_1_grad/Select9^gradients_1/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss/Select_1_grad/Select*'
_output_shapes
:џџџџџџџџџ
В
Bgradients_1/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss/Select_1_grad/Select_19^gradients_1/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*C
_class9
75loc:@gradients_1/logistic_loss/Select_1_grad/Select_1
Џ
,gradients_1/Decoder/last_layer_grad/TanhGradTanhGradDecoder/last_layer.gradients_1/Decoder/reshape_image_grad/Reshape*(
_output_shapes
:џџџџџџџџџ*
T0
Ё
&gradients_1/logistic_loss/Neg_grad/NegNeg@gradients_1/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
2gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/Decoder/last_layer_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ѓ
7gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad-^gradients_1/Decoder/last_layer_grad/TanhGrad
Ї
?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_1/Decoder/last_layer_grad/TanhGrad8^gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_deps*?
_class5
31loc:@gradients_1/Decoder/last_layer_grad/TanhGrad*(
_output_shapes
:џџџџџџџџџ*
T0
Ј
Agradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad8^gradients_1/Decoder/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/Decoder/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ы
,gradients_1/Decoder/dense/MatMul_grad/MatMulMatMul?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependencyDecoder/dense/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ы
.gradients_1/Decoder/dense/MatMul_grad/MatMul_1MatMulDecoder/second_layer/leaky_relu?gradients_1/Decoder/dense/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

6gradients_1/Decoder/dense/MatMul_grad/tuple/group_depsNoOp-^gradients_1/Decoder/dense/MatMul_grad/MatMul/^gradients_1/Decoder/dense/MatMul_grad/MatMul_1
Ѕ
>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/Decoder/dense/MatMul_grad/MatMul7^gradients_1/Decoder/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/Decoder/dense/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
Ѓ
@gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/Decoder/dense/MatMul_grad/MatMul_17^gradients_1/Decoder/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/Decoder/dense/MatMul_grad/MatMul_1* 
_output_shapes
:

§
gradients_1/AddNAddN>gradients_1/logistic_loss/Select_grad/tuple/control_dependency;gradients_1/logistic_loss/mul_grad/tuple/control_dependencyBgradients_1/logistic_loss/Select_1_grad/tuple/control_dependency_1&gradients_1/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:џџџџџџџџџ

7gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
_output_shapes
:*
T0*
data_formatNHWC

<gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN8^gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad

Dgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN=^gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select
Л
Fgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity7gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad=^gradients_1/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@gradients_1/Discriminator/prob/BiasAdd_grad/BiasAddGrad

6gradients_1/Decoder/second_layer/leaky_relu_grad/ShapeShape#Decoder/second_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
А
8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1Shape8Decoder/second_layer/batch_normalization/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
Ж
8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_2Shape>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

<gradients_1/Decoder/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ћ
6gradients_1/Decoder/second_layer/leaky_relu_grad/zerosFill8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_2<gradients_1/Decoder/second_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
п
=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Decoder/second_layer/leaky_relu/mul8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

Fgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Decoder/second_layer/leaky_relu_grad/Shape8gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
7gradients_1/Decoder/second_layer/leaky_relu_grad/SelectSelect=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqual>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency6gradients_1/Decoder/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
­
9gradients_1/Decoder/second_layer/leaky_relu_grad/Select_1Select=gradients_1/Decoder/second_layer/leaky_relu_grad/GreaterEqual6gradients_1/Decoder/second_layer/leaky_relu_grad/zeros>gradients_1/Decoder/dense/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
ќ
4gradients_1/Decoder/second_layer/leaky_relu_grad/SumSum7gradients_1/Decoder/second_layer/leaky_relu_grad/SelectFgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ђ
8gradients_1/Decoder/second_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Decoder/second_layer/leaky_relu_grad/Sum6gradients_1/Decoder/second_layer/leaky_relu_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

6gradients_1/Decoder/second_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Decoder/second_layer/leaky_relu_grad/Select_1Hgradients_1/Decoder/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ј
:gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Decoder/second_layer/leaky_relu_grad/Sum_18gradients_1/Decoder/second_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
С
Agradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape;^gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1
г
Igradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Decoder/second_layer/leaky_relu_grad/ReshapeB^gradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*K
_classA
?=loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape
й
Kgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1B^gradients_1/Decoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
њ
1gradients_1/Discriminator/prob/MatMul_grad/MatMulMatMulDgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
њ
3gradients_1/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluDgradients_1/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
­
;gradients_1/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp2^gradients_1/Discriminator/prob/MatMul_grad/MatMul4^gradients_1/Discriminator/prob/MatMul_grad/MatMul_1
Й
Cgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity1gradients_1/Discriminator/prob/MatMul_grad/MatMul<^gradients_1/Discriminator/prob/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*D
_class:
86loc:@gradients_1/Discriminator/prob/MatMul_grad/MatMul
Ж
Egradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity3gradients_1/Discriminator/prob/MatMul_grad/MatMul_1<^gradients_1/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	
}
:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Д
<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1Shape8Decoder/second_layer/batch_normalization/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0

Jgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ї
8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/MulMulIgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency8Decoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/SumSum8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/MulJgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ь
<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ц
:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Mul_1Mul%Decoder/second_layer/leaky_relu/alphaIgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Decoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Э
Egradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1
б
Mgradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Decoder/second_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*O
_classE
CAloc:@gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape
щ
Ogradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Decoder/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Ѕ
<gradients_1/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
А
>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
С
>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0

Bgradients_1/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

<gradients_1/Discriminator/second_layer/leaky_relu_grad/zerosFill>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_2Bgradients_1/Discriminator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
х
Cgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
 
Lgradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Т
=gradients_1/Discriminator/second_layer/leaky_relu_grad/SelectSelectCgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqualCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency<gradients_1/Discriminator/second_layer/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
Ф
?gradients_1/Discriminator/second_layer/leaky_relu_grad/Select_1SelectCgradients_1/Discriminator/second_layer/leaky_relu_grad/GreaterEqual<gradients_1/Discriminator/second_layer/leaky_relu_grad/zerosCgradients_1/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

:gradients_1/Discriminator/second_layer/leaky_relu_grad/SumSum=gradients_1/Discriminator/second_layer/leaky_relu_grad/SelectLgradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape:gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum<gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

<gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum?gradients_1/Discriminator/second_layer/leaky_relu_grad/Select_1Ngradients_1/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape<gradients_1/Discriminator/second_layer/leaky_relu_grad/Sum_1>gradients_1/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Ggradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp?^gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeA^gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1
ы
Ogradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity>gradients_1/Discriminator/second_layer/leaky_relu_grad/ReshapeH^gradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ё
Qgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1H^gradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ*
T0
У
gradients_1/AddN_1AddNKgradients_1/Decoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Decoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:џџџџџџџџџ*
T0*M
_classC
A?loc:@gradients_1/Decoder/second_layer/leaky_relu_grad/Reshape_1*
N
Ч
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Decoder/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:

Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
й
_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_1_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Н
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_1agradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
Sgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

Zgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
З
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
А
dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1

@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Д
Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ќ
Pgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ShapeBgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
§
>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/MulMulOgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/SumSum>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/MulPgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ў
Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape>gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ј
@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaOgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Rgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Dgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Bgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
п
Kgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
щ
Sgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeL^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 

Ugradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1L^gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Л
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Decoder/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:

Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
й
_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѓ
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Decoder/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
Ф
Mgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Decoder/second_layer/fully_connected/BiasAddbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Ъ
Ogradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
Sgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

Zgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
З
bgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
А
dgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
е
gradients_1/AddN_2AddNQgradients_1/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Ugradients_1/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Џ
Ogradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:
У
Tgradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2P^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
л
\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2U^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer/leaky_relu_grad/Reshape_1

^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradU^gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
љ
Igradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Ngradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
А
Vgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Decoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ

Xgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Т
Igradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMul\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Њ
Kgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_relu\gradients_1/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ѕ
Sgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulL^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1

[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulT^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*\
_classR
PNloc:@gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ*
T0

]gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1T^gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

А
Cgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Decoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

Egradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1MatMulDecoder/first_layer/leaky_reluVgradients_1/Decoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
у
Mgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1

Ugradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
џ
Wgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Decoder/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

Ѓ
;gradients_1/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
Ў
=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
и
=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_2Shape[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Agradients_1/Discriminator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

;gradients_1/Discriminator/first_layer/leaky_relu_grad/zerosFill=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_2Agradients_1/Discriminator/first_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
т
Bgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

Kgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
з
<gradients_1/Discriminator/first_layer/leaky_relu_grad/SelectSelectBgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqual[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency;gradients_1/Discriminator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
й
>gradients_1/Discriminator/first_layer/leaky_relu_grad/Select_1SelectBgradients_1/Discriminator/first_layer/leaky_relu_grad/GreaterEqual;gradients_1/Discriminator/first_layer/leaky_relu_grad/zeros[gradients_1/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients_1/Discriminator/first_layer/leaky_relu_grad/SumSum<gradients_1/Discriminator/first_layer/leaky_relu_grad/SelectKgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

=gradients_1/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape9gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum;gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

;gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum>gradients_1/Discriminator/first_layer/leaky_relu_grad/Select_1Mgradients_1/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

?gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape;gradients_1/Discriminator/first_layer/leaky_relu_grad/Sum_1=gradients_1/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
а
Fgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp>^gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape@^gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1
ч
Ngradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity=gradients_1/Discriminator/first_layer/leaky_relu_grad/ReshapeG^gradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
э
Pgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity?gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1G^gradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

5gradients_1/Decoder/first_layer/leaky_relu_grad/ShapeShape"Decoder/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ђ
7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1Shape+Decoder/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ь
7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

;gradients_1/Decoder/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
5gradients_1/Decoder/first_layer/leaky_relu_grad/zerosFill7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_2;gradients_1/Decoder/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
а
<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual"Decoder/first_layer/leaky_relu/mul+Decoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

Egradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/Decoder/first_layer/leaky_relu_grad/Shape7gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
П
6gradients_1/Decoder/first_layer/leaky_relu_grad/SelectSelect<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqualUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency5gradients_1/Decoder/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
С
8gradients_1/Decoder/first_layer/leaky_relu_grad/Select_1Select<gradients_1/Decoder/first_layer/leaky_relu_grad/GreaterEqual5gradients_1/Decoder/first_layer/leaky_relu_grad/zerosUgradients_1/Decoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
љ
3gradients_1/Decoder/first_layer/leaky_relu_grad/SumSum6gradients_1/Decoder/first_layer/leaky_relu_grad/SelectEgradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
я
7gradients_1/Decoder/first_layer/leaky_relu_grad/ReshapeReshape3gradients_1/Decoder/first_layer/leaky_relu_grad/Sum5gradients_1/Decoder/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
џ
5gradients_1/Decoder/first_layer/leaky_relu_grad/Sum_1Sum8gradients_1/Decoder/first_layer/leaky_relu_grad/Select_1Ggradients_1/Decoder/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ѕ
9gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1Reshape5gradients_1/Decoder/first_layer/leaky_relu_grad/Sum_17gradients_1/Decoder/first_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
О
@gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_depsNoOp8^gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape:^gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1
Я
Hgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity7gradients_1/Decoder/first_layer/leaky_relu_grad/ReshapeA^gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*J
_class@
><loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape
е
Jgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity9gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1A^gradients_1/Decoder/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
В
Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Љ
Ogradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ShapeAgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
њ
=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/MulMulNgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/SumSum=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/MulOgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ћ
Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape=gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ѕ
?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaNgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Qgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Cgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape?gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Agradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
м
Jgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeD^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
х
Rgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeK^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
§
Tgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1K^gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
|
9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
І
;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1Shape+Decoder/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Igradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ш
7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/MulMulHgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency+Decoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/SumSum7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/MulIgradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
щ
;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ReshapeReshape7gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
у
9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Mul_1Mul$Decoder/first_layer/leaky_relu/alphaHgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum_1Sum9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Mul_1Kgradients_1/Decoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

=gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1Reshape9gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Sum_1;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
Ъ
Dgradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp<^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape>^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1
Э
Lgradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity;gradients_1/Decoder/first_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
х
Ngradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity=gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1E^gradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Decoder/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
в
gradients_1/AddN_3AddNPgradients_1/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Tgradients_1/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:џџџџџџџџџ*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N
Ў
Ngradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_3*
data_formatNHWC*
_output_shapes	
:*
T0
С
Sgradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_3O^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
и
[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_3T^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer/leaky_relu_grad/Reshape_1

]gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradT^gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Р
gradients_1/AddN_4AddNJgradients_1/Decoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Ngradients_1/Decoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Ј
Hgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_4*
data_formatNHWC*
_output_shapes	
:*
T0
Е
Mgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_4I^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ц
Ugradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_4N^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/Decoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ*
T0

Wgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradN^gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*[
_classQ
OMloc:@gradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
О
Hgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMul[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(*
T0

Jgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_code[gradients_1/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d*
transpose_a(*
transpose_b( *
T0
ђ
Rgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulK^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1

Zgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulS^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџd

\gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1S^gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d
Ќ
Bgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulMatMulUgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency/Decoder/first_layer/fully_connected/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(

Dgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/encoder_codeUgradients_1/Decoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d*
transpose_a(*
transpose_b( 
р
Lgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpC^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulE^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1
ќ
Tgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityBgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMulM^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџd
њ
Vgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityDgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1M^gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
_output_shapes
:	d*
T0*W
_classM
KIloc:@gradients_1/Decoder/first_layer/fully_connected/MatMul_grad/MatMul_1
ф
gradients_1/AddN_5AddNZgradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyTgradients_1/Decoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*[
_classQ
OMloc:@gradients_1/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*
N*'
_output_shapes
:џџџџџџџџџd

1gradients_1/Encoder/encoder_code_grad/SigmoidGradSigmoidGradEncoder/encoder_codegradients_1/AddN_5*
T0*'
_output_shapes
:џџџџџџџџџd
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
в
2gradients_1/Encoder/Add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients_1/Encoder/Add_grad/Shape$gradients_1/Encoder/Add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ю
 gradients_1/Encoder/Add_grad/SumSum1gradients_1/Encoder/encoder_code_grad/SigmoidGrad2gradients_1/Encoder/Add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Е
$gradients_1/Encoder/Add_grad/ReshapeReshape gradients_1/Encoder/Add_grad/Sum"gradients_1/Encoder/Add_grad/Shape*'
_output_shapes
:џџџџџџџџџd*
T0*
Tshape0
в
"gradients_1/Encoder/Add_grad/Sum_1Sum1gradients_1/Encoder/encoder_code_grad/SigmoidGrad4gradients_1/Encoder/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
&gradients_1/Encoder/Add_grad/Reshape_1Reshape"gradients_1/Encoder/Add_grad/Sum_1$gradients_1/Encoder/Add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџd

-gradients_1/Encoder/Add_grad/tuple/group_depsNoOp%^gradients_1/Encoder/Add_grad/Reshape'^gradients_1/Encoder/Add_grad/Reshape_1

5gradients_1/Encoder/Add_grad/tuple/control_dependencyIdentity$gradients_1/Encoder/Add_grad/Reshape.^gradients_1/Encoder/Add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients_1/Encoder/Add_grad/Reshape*'
_output_shapes
:џџџџџџџџџd

7gradients_1/Encoder/Add_grad/tuple/control_dependency_1Identity&gradients_1/Encoder/Add_grad/Reshape_1.^gradients_1/Encoder/Add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/Encoder/Add_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџd
s
)gradients_1/Encoder/logvar_std_grad/ShapeConst*
valueB:d*
dtype0*
_output_shapes
:
v
+gradients_1/Encoder/logvar_std_grad/Shape_1ShapeEncoder/Exp*
T0*
out_type0*
_output_shapes
:
ч
9gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgsBroadcastGradientArgs)gradients_1/Encoder/logvar_std_grad/Shape+gradients_1/Encoder/logvar_std_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Є
'gradients_1/Encoder/logvar_std_grad/MulMul5gradients_1/Encoder/Add_grad/tuple/control_dependencyEncoder/Exp*'
_output_shapes
:џџџџџџџџџd*
T0
в
'gradients_1/Encoder/logvar_std_grad/SumSum'gradients_1/Encoder/logvar_std_grad/Mul9gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
+gradients_1/Encoder/logvar_std_grad/ReshapeReshape'gradients_1/Encoder/logvar_std_grad/Sum)gradients_1/Encoder/logvar_std_grad/Shape*
T0*
Tshape0*
_output_shapes
:d
А
)gradients_1/Encoder/logvar_std_grad/Mul_1MulEncoder/random_normal5gradients_1/Encoder/Add_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџd*
T0
и
)gradients_1/Encoder/logvar_std_grad/Sum_1Sum)gradients_1/Encoder/logvar_std_grad/Mul_1;gradients_1/Encoder/logvar_std_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
а
-gradients_1/Encoder/logvar_std_grad/Reshape_1Reshape)gradients_1/Encoder/logvar_std_grad/Sum_1+gradients_1/Encoder/logvar_std_grad/Shape_1*'
_output_shapes
:џџџџџџџџџd*
T0*
Tshape0

4gradients_1/Encoder/logvar_std_grad/tuple/group_depsNoOp,^gradients_1/Encoder/logvar_std_grad/Reshape.^gradients_1/Encoder/logvar_std_grad/Reshape_1

<gradients_1/Encoder/logvar_std_grad/tuple/control_dependencyIdentity+gradients_1/Encoder/logvar_std_grad/Reshape5^gradients_1/Encoder/logvar_std_grad/tuple/group_deps*>
_class4
20loc:@gradients_1/Encoder/logvar_std_grad/Reshape*
_output_shapes
:d*
T0
Є
>gradients_1/Encoder/logvar_std_grad/tuple/control_dependency_1Identity-gradients_1/Encoder/logvar_std_grad/Reshape_15^gradients_1/Encoder/logvar_std_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџd*
T0*@
_class6
42loc:@gradients_1/Encoder/logvar_std_grad/Reshape_1
Л
7gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/Encoder/Add_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:d*
T0
И
<gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_depsNoOp8^gradients_1/Encoder/Add_grad/tuple/control_dependency_18^gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad
Е
Dgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/Encoder/Add_grad/tuple/control_dependency_1=^gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_deps*9
_class/
-+loc:@gradients_1/Encoder/Add_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџd*
T0
Л
Fgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency_1Identity7gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad=^gradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/group_deps*
_output_shapes
:d*
T0*J
_class@
><loc:@gradients_1/Encoder/encoder_mu/BiasAdd_grad/BiasAddGrad
І
 gradients_1/Encoder/Exp_grad/mulMul>gradients_1/Encoder/logvar_std_grad/tuple/control_dependency_1Encoder/Exp*'
_output_shapes
:џџџџџџџџџd*
T0
њ
1gradients_1/Encoder/encoder_mu/MatMul_grad/MatMulMatMulDgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependencyEncoder/encoder_mu/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
є
3gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1MatMulEncoder/second_layer/leaky_reluDgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d*
transpose_a(*
transpose_b( *
T0
­
;gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_depsNoOp2^gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul4^gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1
Й
Cgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependencyIdentity1gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul<^gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
Ж
Egradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependency_1Identity3gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1<^gradients_1/Encoder/encoder_mu/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul_1*
_output_shapes
:	d

&gradients_1/Encoder/truediv_grad/ShapeShapeEncoder/encoder_logvar/BiasAdd*
out_type0*
_output_shapes
:*
T0
k
(gradients_1/Encoder/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
о
6gradients_1/Encoder/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/Encoder/truediv_grad/Shape(gradients_1/Encoder/truediv_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

(gradients_1/Encoder/truediv_grad/RealDivRealDiv gradients_1/Encoder/Exp_grad/mulEncoder/truediv/y*
T0*'
_output_shapes
:џџџџџџџџџd
Э
$gradients_1/Encoder/truediv_grad/SumSum(gradients_1/Encoder/truediv_grad/RealDiv6gradients_1/Encoder/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
С
(gradients_1/Encoder/truediv_grad/ReshapeReshape$gradients_1/Encoder/truediv_grad/Sum&gradients_1/Encoder/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџd
}
$gradients_1/Encoder/truediv_grad/NegNegEncoder/encoder_logvar/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџd
 
*gradients_1/Encoder/truediv_grad/RealDiv_1RealDiv$gradients_1/Encoder/truediv_grad/NegEncoder/truediv/y*'
_output_shapes
:џџџџџџџџџd*
T0
І
*gradients_1/Encoder/truediv_grad/RealDiv_2RealDiv*gradients_1/Encoder/truediv_grad/RealDiv_1Encoder/truediv/y*'
_output_shapes
:џџџџџџџџџd*
T0
Ћ
$gradients_1/Encoder/truediv_grad/mulMul gradients_1/Encoder/Exp_grad/mul*gradients_1/Encoder/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:џџџџџџџџџd
Э
&gradients_1/Encoder/truediv_grad/Sum_1Sum$gradients_1/Encoder/truediv_grad/mul8gradients_1/Encoder/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
*gradients_1/Encoder/truediv_grad/Reshape_1Reshape&gradients_1/Encoder/truediv_grad/Sum_1(gradients_1/Encoder/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

1gradients_1/Encoder/truediv_grad/tuple/group_depsNoOp)^gradients_1/Encoder/truediv_grad/Reshape+^gradients_1/Encoder/truediv_grad/Reshape_1

9gradients_1/Encoder/truediv_grad/tuple/control_dependencyIdentity(gradients_1/Encoder/truediv_grad/Reshape2^gradients_1/Encoder/truediv_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/Encoder/truediv_grad/Reshape*'
_output_shapes
:џџџџџџџџџd*
T0

;gradients_1/Encoder/truediv_grad/tuple/control_dependency_1Identity*gradients_1/Encoder/truediv_grad/Reshape_12^gradients_1/Encoder/truediv_grad/tuple/group_deps*
_output_shapes
: *
T0*=
_class3
1/loc:@gradients_1/Encoder/truediv_grad/Reshape_1
С
;gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients_1/Encoder/truediv_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:d
Т
@gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_depsNoOp<^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGrad:^gradients_1/Encoder/truediv_grad/tuple/control_dependency
С
Hgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependencyIdentity9gradients_1/Encoder/truediv_grad/tuple/control_dependencyA^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/Encoder/truediv_grad/Reshape*'
_output_shapes
:џџџџџџџџџd*
T0
Ы
Jgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGradA^gradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/group_deps*N
_classD
B@loc:@gradients_1/Encoder/encoder_logvar/BiasAdd_grad/BiasAddGrad*
_output_shapes
:d*
T0

5gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMulMatMulHgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency"Encoder/encoder_logvar/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ќ
7gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1MatMulEncoder/second_layer/leaky_reluHgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d*
transpose_a(*
transpose_b( 
Й
?gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_depsNoOp6^gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul8^gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1
Щ
Ggradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependencyIdentity5gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul@^gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_deps*H
_class>
<:loc:@gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ*
T0
Ц
Igradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency_1Identity7gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1@^gradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/Encoder/encoder_logvar/MatMul_grad/MatMul_1*
_output_shapes
:	d
Њ
gradients_1/AddN_6AddNCgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependencyGgradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency*
T0*D
_class:
86loc:@gradients_1/Encoder/encoder_mu/MatMul_grad/MatMul*
N*(
_output_shapes
:џџџџџџџџџ

6gradients_1/Encoder/second_layer/leaky_relu_grad/ShapeShape#Encoder/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
А
8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1Shape8Encoder/second_layer/batch_normalization/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0

8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_2Shapegradients_1/AddN_6*
T0*
out_type0*
_output_shapes
:

<gradients_1/Encoder/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ћ
6gradients_1/Encoder/second_layer/leaky_relu_grad/zerosFill8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_2<gradients_1/Encoder/second_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
п
=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Encoder/second_layer/leaky_relu/mul8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

Fgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Encoder/second_layer/leaky_relu_grad/Shape8gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
џ
7gradients_1/Encoder/second_layer/leaky_relu_grad/SelectSelect=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqualgradients_1/AddN_66gradients_1/Encoder/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients_1/Encoder/second_layer/leaky_relu_grad/Select_1Select=gradients_1/Encoder/second_layer/leaky_relu_grad/GreaterEqual6gradients_1/Encoder/second_layer/leaky_relu_grad/zerosgradients_1/AddN_6*
T0*(
_output_shapes
:џџџџџџџџџ
ќ
4gradients_1/Encoder/second_layer/leaky_relu_grad/SumSum7gradients_1/Encoder/second_layer/leaky_relu_grad/SelectFgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ђ
8gradients_1/Encoder/second_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Encoder/second_layer/leaky_relu_grad/Sum6gradients_1/Encoder/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

6gradients_1/Encoder/second_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Encoder/second_layer/leaky_relu_grad/Select_1Hgradients_1/Encoder/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ј
:gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Encoder/second_layer/leaky_relu_grad/Sum_18gradients_1/Encoder/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
С
Agradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape;^gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1
г
Igradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Encoder/second_layer/leaky_relu_grad/ReshapeB^gradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
й
Kgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1B^gradients_1/Encoder/second_layer/leaky_relu_grad/tuple/group_deps*M
_classC
A?loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ*
T0
}
:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Д
<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1Shape8Encoder/second_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0

Jgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ї
8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/MulMulIgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency8Encoder/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/SumSum8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/MulJgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ь
<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
ц
:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Mul_1Mul%Encoder/second_layer/leaky_relu/alphaIgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Encoder/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Э
Egradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1
б
Mgradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Encoder/second_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
щ
Ogradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Encoder/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
У
gradients_1/AddN_7AddNKgradients_1/Encoder/second_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Encoder/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Encoder/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Ч
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Encoder/second_layer/batch_normalization/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0

Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
й
_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_7_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Н
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_7agradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
Sgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
З
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
А
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Л
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Encoder/second_layer/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0

Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
й
_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѓ
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Encoder/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:џџџџџџџџџ*
T0
Ф
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0

Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Encoder/second_layer/fully_connected/BiasAddbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Ъ
Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ж
Sgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
З
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ*
T0
А
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
о
Kgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Xgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg
Л
`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:

bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*^
_classT
RPloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/Neg
љ
Igradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:*
T0

Ngradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
А
Vgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*d
_classZ
XVloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ*
T0

Xgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Encoder/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:*
T0

Ogradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Encoder/second_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:

Zgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Ђ
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul
Ј
dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:*
T0*b
_classX
VTloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
А
Cgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Encoder/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

Egradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/first_layer/leaky_reluVgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
у
Mgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1

Ugradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*V
_classL
JHloc:@gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul
џ
Wgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*X
_classN
LJloc:@gradients_1/Encoder/second_layer/fully_connected/MatMul_grad/MatMul_1
§
gradients_1/AddN_8AddNdgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*f
_class\
ZXloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:*
T0
С
Kgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_83Encoder/second_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
Ш
Mgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_88Encoder/second_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:
ў
Xgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1

`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:
 
bgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:

5gradients_1/Encoder/first_layer/leaky_relu_grad/ShapeShape"Encoder/first_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
Ђ
7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1Shape+Encoder/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ь
7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

;gradients_1/Encoder/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ј
5gradients_1/Encoder/first_layer/leaky_relu_grad/zerosFill7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_2;gradients_1/Encoder/first_layer/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:џџџџџџџџџ*
T0
а
<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual"Encoder/first_layer/leaky_relu/mul+Encoder/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

Egradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs5gradients_1/Encoder/first_layer/leaky_relu_grad/Shape7gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
П
6gradients_1/Encoder/first_layer/leaky_relu_grad/SelectSelect<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqualUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency5gradients_1/Encoder/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
С
8gradients_1/Encoder/first_layer/leaky_relu_grad/Select_1Select<gradients_1/Encoder/first_layer/leaky_relu_grad/GreaterEqual5gradients_1/Encoder/first_layer/leaky_relu_grad/zerosUgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
љ
3gradients_1/Encoder/first_layer/leaky_relu_grad/SumSum6gradients_1/Encoder/first_layer/leaky_relu_grad/SelectEgradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
я
7gradients_1/Encoder/first_layer/leaky_relu_grad/ReshapeReshape3gradients_1/Encoder/first_layer/leaky_relu_grad/Sum5gradients_1/Encoder/first_layer/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
џ
5gradients_1/Encoder/first_layer/leaky_relu_grad/Sum_1Sum8gradients_1/Encoder/first_layer/leaky_relu_grad/Select_1Ggradients_1/Encoder/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ѕ
9gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1Reshape5gradients_1/Encoder/first_layer/leaky_relu_grad/Sum_17gradients_1/Encoder/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
О
@gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_depsNoOp8^gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape:^gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1
Я
Hgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity7gradients_1/Encoder/first_layer/leaky_relu_grad/ReshapeA^gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*J
_class@
><loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape
е
Jgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity9gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1A^gradients_1/Encoder/first_layer/leaky_relu_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ*
T0
|
9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
І
;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1Shape+Encoder/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:

Igradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ш
7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/MulMulHgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency+Encoder/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/SumSum7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/MulIgradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
щ
;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ReshapeReshape7gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
у
9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Mul_1Mul$Encoder/first_layer/leaky_relu/alphaHgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum_1Sum9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Mul_1Kgradients_1/Encoder/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

=gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1Reshape9gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Sum_1;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ъ
Dgradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp<^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape>^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1
Э
Lgradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity;gradients_1/Encoder/first_layer/leaky_relu/mul_grad/ReshapeE^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
х
Ngradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity=gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1E^gradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Encoder/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Р
gradients_1/AddN_9AddNJgradients_1/Encoder/first_layer/leaky_relu_grad/tuple/control_dependency_1Ngradients_1/Encoder/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Ј
Hgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
_output_shapes	
:*
T0*
data_formatNHWC
Е
Mgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_9I^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ц
Ugradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_9N^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*L
_classB
@>loc:@gradients_1/Encoder/first_layer/leaky_relu_grad/Reshape_1

Wgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGradN^gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*[
_classQ
OMloc:@gradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
­
Bgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulMatMulUgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency/Encoder/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

Dgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulEncoder/ReshapeUgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
р
Lgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpC^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulE^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1
§
Tgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityBgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMulM^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*U
_classK
IGloc:@gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul
ћ
Vgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityDgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1M^gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*W
_classM
KIloc:@gradients_1/Encoder/first_layer/fully_connected/MatMul_grad/MatMul_1

beta1_power_1/initial_valueConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ё
beta1_power_1
VariableV2*.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Ф
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
~
beta1_power_1/readIdentitybeta1_power_1*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
: 

beta2_power_1/initial_valueConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueB
 *wО?*
dtype0*
_output_shapes
: 
Ё
beta2_power_1
VariableV2*.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Ф
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
: 
~
beta2_power_1/readIdentitybeta2_power_1*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
: *
T0
ы
[Encoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
е
QEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*
_output_shapes
: *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
џ
KEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill[Encoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorQEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const* 
_output_shapes
:
*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*

index_type0
№
9Encoder/first_layer/fully_connected/kernel/generator_opti
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container *
shape:

х
@Encoder/first_layer/fully_connected/kernel/generator_opti/AssignAssign9Encoder/first_layer/fully_connected/kernel/generator_optiKEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

я
>Encoder/first_layer/fully_connected/kernel/generator_opti/readIdentity9Encoder/first_layer/fully_connected/kernel/generator_opti*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:

э
]Encoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
з
SEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

MEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill]Encoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorSEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
*
T0
ђ
;Encoder/first_layer/fully_connected/kernel/generator_opti_1
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
	container 
ы
BEncoder/first_layer/fully_connected/kernel/generator_opti_1/AssignAssign;Encoder/first_layer/fully_connected/kernel/generator_opti_1MEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
validate_shape(
ѓ
@Encoder/first_layer/fully_connected/kernel/generator_opti_1/readIdentity;Encoder/first_layer/fully_connected/kernel/generator_opti_1*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
е
IEncoder/first_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*
_output_shapes	
:*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB*    *
dtype0
т
7Encoder/first_layer/fully_connected/bias/generator_opti
VariableV2*
shared_name *;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
и
>Encoder/first_layer/fully_connected/bias/generator_opti/AssignAssign7Encoder/first_layer/fully_connected/bias/generator_optiIEncoder/first_layer/fully_connected/bias/generator_opti/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ф
<Encoder/first_layer/fully_connected/bias/generator_opti/readIdentity7Encoder/first_layer/fully_connected/bias/generator_opti*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:
з
KEncoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ф
9Encoder/first_layer/fully_connected/bias/generator_opti_1
VariableV2*
shared_name *;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
о
@Encoder/first_layer/fully_connected/bias/generator_opti_1/AssignAssign9Encoder/first_layer/fully_connected/bias/generator_opti_1KEncoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ш
>Encoder/first_layer/fully_connected/bias/generator_opti_1/readIdentity9Encoder/first_layer/fully_connected/bias/generator_opti_1*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
_output_shapes	
:*
T0
э
\Encoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
з
REncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/ConstConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

LEncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zerosFill\Encoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/shape_as_tensorREncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros/Const* 
_output_shapes
:
*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*

index_type0
ђ
:Encoder/second_layer/fully_connected/kernel/generator_opti
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:

щ
AEncoder/second_layer/fully_connected/kernel/generator_opti/AssignAssign:Encoder/second_layer/fully_connected/kernel/generator_optiLEncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

ђ
?Encoder/second_layer/fully_connected/kernel/generator_opti/readIdentity:Encoder/second_layer/fully_connected/kernel/generator_opti*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:

я
^Encoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
й
TEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

NEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zerosFill^Encoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorTEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*

index_type0
є
<Encoder/second_layer/fully_connected/kernel/generator_opti_1
VariableV2*
shared_name *>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

я
CEncoder/second_layer/fully_connected/kernel/generator_opti_1/AssignAssign<Encoder/second_layer/fully_connected/kernel/generator_opti_1NEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

і
AEncoder/second_layer/fully_connected/kernel/generator_opti_1/readIdentity<Encoder/second_layer/fully_connected/kernel/generator_opti_1*
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel* 
_output_shapes
:

з
JEncoder/second_layer/fully_connected/bias/generator_opti/Initializer/zerosConst*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ф
8Encoder/second_layer/fully_connected/bias/generator_opti
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias
м
?Encoder/second_layer/fully_connected/bias/generator_opti/AssignAssign8Encoder/second_layer/fully_connected/bias/generator_optiJEncoder/second_layer/fully_connected/bias/generator_opti/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ч
=Encoder/second_layer/fully_connected/bias/generator_opti/readIdentity8Encoder/second_layer/fully_connected/bias/generator_opti*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
_output_shapes	
:
й
LEncoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zerosConst*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ц
:Encoder/second_layer/fully_connected/bias/generator_opti_1
VariableV2*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
т
AEncoder/second_layer/fully_connected/bias/generator_opti_1/AssignAssign:Encoder/second_layer/fully_connected/bias/generator_opti_1LEncoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
validate_shape(
ы
?Encoder/second_layer/fully_connected/bias/generator_opti_1/readIdentity:Encoder/second_layer/fully_connected/bias/generator_opti_1*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
_output_shapes	
:*
T0
с
OEncoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zerosConst*
_output_shapes	
:*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB*    *
dtype0
ю
=Encoder/second_layer/batch_normalization/gamma/generator_opti
VariableV2*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
№
DEncoder/second_layer/batch_normalization/gamma/generator_opti/AssignAssign=Encoder/second_layer/batch_normalization/gamma/generator_optiOEncoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
і
BEncoder/second_layer/batch_normalization/gamma/generator_opti/readIdentity=Encoder/second_layer/batch_normalization/gamma/generator_opti*
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:
у
QEncoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zerosConst*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
№
?Encoder/second_layer/batch_normalization/gamma/generator_opti_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
	container *
shape:
і
FEncoder/second_layer/batch_normalization/gamma/generator_opti_1/AssignAssign?Encoder/second_layer/batch_normalization/gamma/generator_opti_1QEncoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zeros*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
њ
DEncoder/second_layer/batch_normalization/gamma/generator_opti_1/readIdentity?Encoder/second_layer/batch_normalization/gamma/generator_opti_1*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
_output_shapes	
:*
T0
п
NEncoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zerosConst*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ь
<Encoder/second_layer/batch_normalization/beta/generator_opti
VariableV2*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ь
CEncoder/second_layer/batch_normalization/beta/generator_opti/AssignAssign<Encoder/second_layer/batch_normalization/beta/generator_optiNEncoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zeros*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ѓ
AEncoder/second_layer/batch_normalization/beta/generator_opti/readIdentity<Encoder/second_layer/batch_normalization/beta/generator_opti*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:
с
PEncoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zerosConst*
_output_shapes	
:*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
valueB*    *
dtype0
ю
>Encoder/second_layer/batch_normalization/beta/generator_opti_1
VariableV2*
shared_name *@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ђ
EEncoder/second_layer/batch_normalization/beta/generator_opti_1/AssignAssign>Encoder/second_layer/batch_normalization/beta/generator_opti_1PEncoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
validate_shape(
ї
CEncoder/second_layer/batch_normalization/beta/generator_opti_1/readIdentity>Encoder/second_layer/batch_normalization/beta/generator_opti_1*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
_output_shapes	
:*
T0
Щ
JEncoder/encoder_mu/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
Г
@Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros/ConstConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
К
:Encoder/encoder_mu/kernel/generator_opti/Initializer/zerosFillJEncoder/encoder_mu/kernel/generator_opti/Initializer/zeros/shape_as_tensor@Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros/Const*
_output_shapes
:	d*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*

index_type0
Ь
(Encoder/encoder_mu/kernel/generator_opti
VariableV2*
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name *,
_class"
 loc:@Encoder/encoder_mu/kernel*
	container 
 
/Encoder/encoder_mu/kernel/generator_opti/AssignAssign(Encoder/encoder_mu/kernel/generator_opti:Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0
Л
-Encoder/encoder_mu/kernel/generator_opti/readIdentity(Encoder/encoder_mu/kernel/generator_opti*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel*
_output_shapes
:	d
Ы
LEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB"   d   *
dtype0*
_output_shapes
:
Е
BEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/ConstConst*,
_class"
 loc:@Encoder/encoder_mu/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
<Encoder/encoder_mu/kernel/generator_opti_1/Initializer/zerosFillLEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorBEncoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros/Const*,
_class"
 loc:@Encoder/encoder_mu/kernel*

index_type0*
_output_shapes
:	d*
T0
Ю
*Encoder/encoder_mu/kernel/generator_opti_1
VariableV2*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name *,
_class"
 loc:@Encoder/encoder_mu/kernel
І
1Encoder/encoder_mu/kernel/generator_opti_1/AssignAssign*Encoder/encoder_mu/kernel/generator_opti_1<Encoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros*,
_class"
 loc:@Encoder/encoder_mu/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0
П
/Encoder/encoder_mu/kernel/generator_opti_1/readIdentity*Encoder/encoder_mu/kernel/generator_opti_1*
_output_shapes
:	d*
T0*,
_class"
 loc:@Encoder/encoder_mu/kernel
Б
8Encoder/encoder_mu/bias/generator_opti/Initializer/zerosConst**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0*
_output_shapes
:d
О
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

-Encoder/encoder_mu/bias/generator_opti/AssignAssign&Encoder/encoder_mu/bias/generator_opti8Encoder/encoder_mu/bias/generator_opti/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
validate_shape(*
_output_shapes
:d
А
+Encoder/encoder_mu/bias/generator_opti/readIdentity&Encoder/encoder_mu/bias/generator_opti*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d
Г
:Encoder/encoder_mu/bias/generator_opti_1/Initializer/zerosConst**
_class 
loc:@Encoder/encoder_mu/bias*
valueBd*    *
dtype0*
_output_shapes
:d
Р
(Encoder/encoder_mu/bias/generator_opti_1
VariableV2*
_output_shapes
:d*
shared_name **
_class 
loc:@Encoder/encoder_mu/bias*
	container *
shape:d*
dtype0

/Encoder/encoder_mu/bias/generator_opti_1/AssignAssign(Encoder/encoder_mu/bias/generator_opti_1:Encoder/encoder_mu/bias/generator_opti_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Encoder/encoder_mu/bias*
validate_shape(*
_output_shapes
:d
Д
-Encoder/encoder_mu/bias/generator_opti_1/readIdentity(Encoder/encoder_mu/bias/generator_opti_1**
_class 
loc:@Encoder/encoder_mu/bias*
_output_shapes
:d*
T0
б
NEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0
Л
DEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/ConstConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ъ
>Encoder/encoder_logvar/kernel/generator_opti/Initializer/zerosFillNEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/shape_as_tensorDEncoder/encoder_logvar/kernel/generator_opti/Initializer/zeros/Const*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*

index_type0*
_output_shapes
:	d
д
,Encoder/encoder_logvar/kernel/generator_opti
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	d
А
3Encoder/encoder_logvar/kernel/generator_opti/AssignAssign,Encoder/encoder_logvar/kernel/generator_opti>Encoder/encoder_logvar/kernel/generator_opti/Initializer/zeros*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0
Ч
1Encoder/encoder_logvar/kernel/generator_opti/readIdentity,Encoder/encoder_logvar/kernel/generator_opti*
_output_shapes
:	d*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel
г
PEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB"   d   *
dtype0
Н
FEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/ConstConst*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
а
@Encoder/encoder_logvar/kernel/generator_opti_1/Initializer/zerosFillPEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/shape_as_tensorFEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros/Const*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*

index_type0*
_output_shapes
:	d*
T0
ж
.Encoder/encoder_logvar/kernel/generator_opti_1
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
	container *
shape:	d
Ж
5Encoder/encoder_logvar/kernel/generator_opti_1/AssignAssign.Encoder/encoder_logvar/kernel/generator_opti_1@Encoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0
Ы
3Encoder/encoder_logvar/kernel/generator_opti_1/readIdentity.Encoder/encoder_logvar/kernel/generator_opti_1*
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
_output_shapes
:	d
Й
<Encoder/encoder_logvar/bias/generator_opti/Initializer/zerosConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0*
_output_shapes
:d
Ц
*Encoder/encoder_logvar/bias/generator_opti
VariableV2*
shape:d*
dtype0*
_output_shapes
:d*
shared_name *.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container 
Ѓ
1Encoder/encoder_logvar/bias/generator_opti/AssignAssign*Encoder/encoder_logvar/bias/generator_opti<Encoder/encoder_logvar/bias/generator_opti/Initializer/zeros*
_output_shapes
:d*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(
М
/Encoder/encoder_logvar/bias/generator_opti/readIdentity*Encoder/encoder_logvar/bias/generator_opti*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
:d
Л
>Encoder/encoder_logvar/bias/generator_opti_1/Initializer/zerosConst*.
_class$
" loc:@Encoder/encoder_logvar/bias*
valueBd*    *
dtype0*
_output_shapes
:d
Ш
,Encoder/encoder_logvar/bias/generator_opti_1
VariableV2*
_output_shapes
:d*
shared_name *.
_class$
" loc:@Encoder/encoder_logvar/bias*
	container *
shape:d*
dtype0
Љ
3Encoder/encoder_logvar/bias/generator_opti_1/AssignAssign,Encoder/encoder_logvar/bias/generator_opti_1>Encoder/encoder_logvar/bias/generator_opti_1/Initializer/zeros*
_output_shapes
:d*
use_locking(*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(
Р
1Encoder/encoder_logvar/bias/generator_opti_1/readIdentity,Encoder/encoder_logvar/bias/generator_opti_1*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
:d
a
generator_opti/learning_rateConst*
valueB
 *ЗQ9*
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
 *wО?*
dtype0*
_output_shapes
: 
[
generator_opti/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
ю
Jgenerator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam*Encoder/first_layer/fully_connected/kernel9Encoder/first_layer/fully_connected/kernel/generator_opti;Encoder/first_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonVgradients_1/Encoder/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*=
_class3
1/loc:@Encoder/first_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
р
Hgenerator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam(Encoder/first_layer/fully_connected/bias7Encoder/first_layer/fully_connected/bias/generator_opti9Encoder/first_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Encoder/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*;
_class1
/-loc:@Encoder/first_layer/fully_connected/bias*
use_nesterov( 
є
Kgenerator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Encoder/second_layer/fully_connected/kernel:Encoder/second_layer/fully_connected/kernel/generator_opti<Encoder/second_layer/fully_connected/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonWgradients_1/Encoder/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*>
_class4
20loc:@Encoder/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:

ц
Igenerator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Encoder/second_layer/fully_connected/bias8Encoder/second_layer/fully_connected/bias/generator_opti:Encoder/second_layer/fully_connected/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonXgradients_1/Encoder/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*<
_class2
0.loc:@Encoder/second_layer/fully_connected/bias*
use_nesterov( 

Ngenerator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Encoder/second_layer/batch_normalization/gamma=Encoder/second_layer/batch_normalization/gamma/generator_opti?Encoder/second_layer/batch_normalization/gamma/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonbgradients_1/Encoder/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*A
_class7
53loc:@Encoder/second_layer/batch_normalization/gamma*
use_nesterov( 

Mgenerator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Encoder/second_layer/batch_normalization/beta<Encoder/second_layer/batch_normalization/beta/generator_opti>Encoder/second_layer/batch_normalization/beta/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilon`gradients_1/Encoder/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes	
:*
use_locking( *
T0*@
_class6
42loc:@Encoder/second_layer/batch_normalization/beta*
use_nesterov( 

9generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdam	ApplyAdamEncoder/encoder_mu/kernel(Encoder/encoder_mu/kernel/generator_opti*Encoder/encoder_mu/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonEgradients_1/Encoder/encoder_mu/MatMul_grad/tuple/control_dependency_1*,
_class"
 loc:@Encoder/encoder_mu/kernel*
use_nesterov( *
_output_shapes
:	d*
use_locking( *
T0
љ
7generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam	ApplyAdamEncoder/encoder_mu/bias&Encoder/encoder_mu/bias/generator_opti(Encoder/encoder_mu/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonFgradients_1/Encoder/encoder_mu/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:d*
use_locking( *
T0**
_class 
loc:@Encoder/encoder_mu/bias*
use_nesterov( 

=generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam	ApplyAdamEncoder/encoder_logvar/kernel,Encoder/encoder_logvar/kernel/generator_opti.Encoder/encoder_logvar/kernel/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonIgradients_1/Encoder/encoder_logvar/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@Encoder/encoder_logvar/kernel*
use_nesterov( *
_output_shapes
:	d

;generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam	ApplyAdamEncoder/encoder_logvar/bias*Encoder/encoder_logvar/bias/generator_opti,Encoder/encoder_logvar/bias/generator_opti_1beta1_power_1/readbeta2_power_1/readgenerator_opti/learning_rategenerator_opti/beta1generator_opti/beta2generator_opti/epsilonJgradients_1/Encoder/encoder_logvar/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:d*
use_locking( *
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
use_nesterov( 
л
generator_opti/mulMulbeta1_power_1/readgenerator_opti/beta1<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
: 
М
generator_opti/AssignAssignbeta1_power_1generator_opti/mul*
use_locking( *
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(*
_output_shapes
: 
н
generator_opti/mul_1Mulbeta2_power_1/readgenerator_opti/beta2<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam*
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
_output_shapes
: 
Р
generator_opti/Assign_1Assignbeta2_power_1generator_opti/mul_1*
_output_shapes
: *
use_locking( *
T0*.
_class$
" loc:@Encoder/encoder_logvar/bias*
validate_shape(

generator_optiNoOp^generator_opti/Assign^generator_opti/Assign_1<^generator_opti/update_Encoder/encoder_logvar/bias/ApplyAdam>^generator_opti/update_Encoder/encoder_logvar/kernel/ApplyAdam8^generator_opti/update_Encoder/encoder_mu/bias/ApplyAdam:^generator_opti/update_Encoder/encoder_mu/kernel/ApplyAdamI^generator_opti/update_Encoder/first_layer/fully_connected/bias/ApplyAdamK^generator_opti/update_Encoder/first_layer/fully_connected/kernel/ApplyAdamN^generator_opti/update_Encoder/second_layer/batch_normalization/beta/ApplyAdamO^generator_opti/update_Encoder/second_layer/batch_normalization/gamma/ApplyAdamJ^generator_opti/update_Encoder/second_layer/fully_connected/bias/ApplyAdamL^generator_opti/update_Encoder/second_layer/fully_connected/kernel/ApplyAdam
i
Merge/MergeSummaryMergeSummarygenerator_loss_1discriminator_loss*
_output_shapes
: *
N""9
	summaries,
*
generator_loss_1:0
discriminator_loss:0"&
trainable_variablesъ%ч%
п
,Encoder/first_layer/fully_connected/kernel:01Encoder/first_layer/fully_connected/kernel/Assign1Encoder/first_layer/fully_connected/kernel/read:02GEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform:08
Ю
*Encoder/first_layer/fully_connected/bias:0/Encoder/first_layer/fully_connected/bias/Assign/Encoder/first_layer/fully_connected/bias/read:02<Encoder/first_layer/fully_connected/bias/Initializer/zeros:08
у
-Encoder/second_layer/fully_connected/kernel:02Encoder/second_layer/fully_connected/kernel/Assign2Encoder/second_layer/fully_connected/kernel/read:02HEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform:08
в
+Encoder/second_layer/fully_connected/bias:00Encoder/second_layer/fully_connected/bias/Assign0Encoder/second_layer/fully_connected/bias/read:02=Encoder/second_layer/fully_connected/bias/Initializer/zeros:08
х
0Encoder/second_layer/batch_normalization/gamma:05Encoder/second_layer/batch_normalization/gamma/Assign5Encoder/second_layer/batch_normalization/gamma/read:02AEncoder/second_layer/batch_normalization/gamma/Initializer/ones:08
т
/Encoder/second_layer/batch_normalization/beta:04Encoder/second_layer/batch_normalization/beta/Assign4Encoder/second_layer/batch_normalization/beta/read:02AEncoder/second_layer/batch_normalization/beta/Initializer/zeros:08

Encoder/encoder_mu/kernel:0 Encoder/encoder_mu/kernel/Assign Encoder/encoder_mu/kernel/read:026Encoder/encoder_mu/kernel/Initializer/random_uniform:08

Encoder/encoder_mu/bias:0Encoder/encoder_mu/bias/AssignEncoder/encoder_mu/bias/read:02+Encoder/encoder_mu/bias/Initializer/zeros:08
Ћ
Encoder/encoder_logvar/kernel:0$Encoder/encoder_logvar/kernel/Assign$Encoder/encoder_logvar/kernel/read:02:Encoder/encoder_logvar/kernel/Initializer/random_uniform:08

Encoder/encoder_logvar/bias:0"Encoder/encoder_logvar/bias/Assign"Encoder/encoder_logvar/bias/read:02/Encoder/encoder_logvar/bias/Initializer/zeros:08
п
,Decoder/first_layer/fully_connected/kernel:01Decoder/first_layer/fully_connected/kernel/Assign1Decoder/first_layer/fully_connected/kernel/read:02GDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform:08
Ю
*Decoder/first_layer/fully_connected/bias:0/Decoder/first_layer/fully_connected/bias/Assign/Decoder/first_layer/fully_connected/bias/read:02<Decoder/first_layer/fully_connected/bias/Initializer/zeros:08
у
-Decoder/second_layer/fully_connected/kernel:02Decoder/second_layer/fully_connected/kernel/Assign2Decoder/second_layer/fully_connected/kernel/read:02HDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform:08
в
+Decoder/second_layer/fully_connected/bias:00Decoder/second_layer/fully_connected/bias/Assign0Decoder/second_layer/fully_connected/bias/read:02=Decoder/second_layer/fully_connected/bias/Initializer/zeros:08
х
0Decoder/second_layer/batch_normalization/gamma:05Decoder/second_layer/batch_normalization/gamma/Assign5Decoder/second_layer/batch_normalization/gamma/read:02ADecoder/second_layer/batch_normalization/gamma/Initializer/ones:08
т
/Decoder/second_layer/batch_normalization/beta:04Decoder/second_layer/batch_normalization/beta/Assign4Decoder/second_layer/batch_normalization/beta/read:02ADecoder/second_layer/batch_normalization/beta/Initializer/zeros:08

Decoder/dense/kernel:0Decoder/dense/kernel/AssignDecoder/dense/kernel/read:021Decoder/dense/kernel/Initializer/random_uniform:08
v
Decoder/dense/bias:0Decoder/dense/bias/AssignDecoder/dense/bias/read:02&Decoder/dense/bias/Initializer/zeros:08
ї
2Discriminator/first_layer/fully_connected/kernel:07Discriminator/first_layer/fully_connected/kernel/Assign7Discriminator/first_layer/fully_connected/kernel/read:02MDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
ц
0Discriminator/first_layer/fully_connected/bias:05Discriminator/first_layer/fully_connected/bias/Assign5Discriminator/first_layer/fully_connected/bias/read:02BDiscriminator/first_layer/fully_connected/bias/Initializer/zeros:08
ћ
3Discriminator/second_layer/fully_connected/kernel:08Discriminator/second_layer/fully_connected/kernel/Assign8Discriminator/second_layer/fully_connected/kernel/read:02NDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
ъ
1Discriminator/second_layer/fully_connected/bias:06Discriminator/second_layer/fully_connected/bias/Assign6Discriminator/second_layer/fully_connected/bias/read:02CDiscriminator/second_layer/fully_connected/bias/Initializer/zeros:08

Discriminator/prob/kernel:0 Discriminator/prob/kernel/Assign Discriminator/prob/kernel/read:026Discriminator/prob/kernel/Initializer/random_uniform:08

Discriminator/prob/bias:0Discriminator/prob/bias/AssignDiscriminator/prob/bias/read:02+Discriminator/prob/bias/Initializer/zeros:08"2
train_op&
$
discriminator_opti
generator_opti"вs
	variablesФsСs
п
,Encoder/first_layer/fully_connected/kernel:01Encoder/first_layer/fully_connected/kernel/Assign1Encoder/first_layer/fully_connected/kernel/read:02GEncoder/first_layer/fully_connected/kernel/Initializer/random_uniform:08
Ю
*Encoder/first_layer/fully_connected/bias:0/Encoder/first_layer/fully_connected/bias/Assign/Encoder/first_layer/fully_connected/bias/read:02<Encoder/first_layer/fully_connected/bias/Initializer/zeros:08
у
-Encoder/second_layer/fully_connected/kernel:02Encoder/second_layer/fully_connected/kernel/Assign2Encoder/second_layer/fully_connected/kernel/read:02HEncoder/second_layer/fully_connected/kernel/Initializer/random_uniform:08
в
+Encoder/second_layer/fully_connected/bias:00Encoder/second_layer/fully_connected/bias/Assign0Encoder/second_layer/fully_connected/bias/read:02=Encoder/second_layer/fully_connected/bias/Initializer/zeros:08
х
0Encoder/second_layer/batch_normalization/gamma:05Encoder/second_layer/batch_normalization/gamma/Assign5Encoder/second_layer/batch_normalization/gamma/read:02AEncoder/second_layer/batch_normalization/gamma/Initializer/ones:08
т
/Encoder/second_layer/batch_normalization/beta:04Encoder/second_layer/batch_normalization/beta/Assign4Encoder/second_layer/batch_normalization/beta/read:02AEncoder/second_layer/batch_normalization/beta/Initializer/zeros:08
ќ
6Encoder/second_layer/batch_normalization/moving_mean:0;Encoder/second_layer/batch_normalization/moving_mean/Assign;Encoder/second_layer/batch_normalization/moving_mean/read:02HEncoder/second_layer/batch_normalization/moving_mean/Initializer/zeros:0

:Encoder/second_layer/batch_normalization/moving_variance:0?Encoder/second_layer/batch_normalization/moving_variance/Assign?Encoder/second_layer/batch_normalization/moving_variance/read:02KEncoder/second_layer/batch_normalization/moving_variance/Initializer/ones:0

Encoder/encoder_mu/kernel:0 Encoder/encoder_mu/kernel/Assign Encoder/encoder_mu/kernel/read:026Encoder/encoder_mu/kernel/Initializer/random_uniform:08

Encoder/encoder_mu/bias:0Encoder/encoder_mu/bias/AssignEncoder/encoder_mu/bias/read:02+Encoder/encoder_mu/bias/Initializer/zeros:08
Ћ
Encoder/encoder_logvar/kernel:0$Encoder/encoder_logvar/kernel/Assign$Encoder/encoder_logvar/kernel/read:02:Encoder/encoder_logvar/kernel/Initializer/random_uniform:08

Encoder/encoder_logvar/bias:0"Encoder/encoder_logvar/bias/Assign"Encoder/encoder_logvar/bias/read:02/Encoder/encoder_logvar/bias/Initializer/zeros:08
п
,Decoder/first_layer/fully_connected/kernel:01Decoder/first_layer/fully_connected/kernel/Assign1Decoder/first_layer/fully_connected/kernel/read:02GDecoder/first_layer/fully_connected/kernel/Initializer/random_uniform:08
Ю
*Decoder/first_layer/fully_connected/bias:0/Decoder/first_layer/fully_connected/bias/Assign/Decoder/first_layer/fully_connected/bias/read:02<Decoder/first_layer/fully_connected/bias/Initializer/zeros:08
у
-Decoder/second_layer/fully_connected/kernel:02Decoder/second_layer/fully_connected/kernel/Assign2Decoder/second_layer/fully_connected/kernel/read:02HDecoder/second_layer/fully_connected/kernel/Initializer/random_uniform:08
в
+Decoder/second_layer/fully_connected/bias:00Decoder/second_layer/fully_connected/bias/Assign0Decoder/second_layer/fully_connected/bias/read:02=Decoder/second_layer/fully_connected/bias/Initializer/zeros:08
х
0Decoder/second_layer/batch_normalization/gamma:05Decoder/second_layer/batch_normalization/gamma/Assign5Decoder/second_layer/batch_normalization/gamma/read:02ADecoder/second_layer/batch_normalization/gamma/Initializer/ones:08
т
/Decoder/second_layer/batch_normalization/beta:04Decoder/second_layer/batch_normalization/beta/Assign4Decoder/second_layer/batch_normalization/beta/read:02ADecoder/second_layer/batch_normalization/beta/Initializer/zeros:08
ќ
6Decoder/second_layer/batch_normalization/moving_mean:0;Decoder/second_layer/batch_normalization/moving_mean/Assign;Decoder/second_layer/batch_normalization/moving_mean/read:02HDecoder/second_layer/batch_normalization/moving_mean/Initializer/zeros:0

:Decoder/second_layer/batch_normalization/moving_variance:0?Decoder/second_layer/batch_normalization/moving_variance/Assign?Decoder/second_layer/batch_normalization/moving_variance/read:02KDecoder/second_layer/batch_normalization/moving_variance/Initializer/ones:0

Decoder/dense/kernel:0Decoder/dense/kernel/AssignDecoder/dense/kernel/read:021Decoder/dense/kernel/Initializer/random_uniform:08
v
Decoder/dense/bias:0Decoder/dense/bias/AssignDecoder/dense/bias/read:02&Decoder/dense/bias/Initializer/zeros:08
ї
2Discriminator/first_layer/fully_connected/kernel:07Discriminator/first_layer/fully_connected/kernel/Assign7Discriminator/first_layer/fully_connected/kernel/read:02MDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
ц
0Discriminator/first_layer/fully_connected/bias:05Discriminator/first_layer/fully_connected/bias/Assign5Discriminator/first_layer/fully_connected/bias/read:02BDiscriminator/first_layer/fully_connected/bias/Initializer/zeros:08
ћ
3Discriminator/second_layer/fully_connected/kernel:08Discriminator/second_layer/fully_connected/kernel/Assign8Discriminator/second_layer/fully_connected/kernel/read:02NDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
ъ
1Discriminator/second_layer/fully_connected/bias:06Discriminator/second_layer/fully_connected/bias/Assign6Discriminator/second_layer/fully_connected/bias/read:02CDiscriminator/second_layer/fully_connected/bias/Initializer/zeros:08

Discriminator/prob/kernel:0 Discriminator/prob/kernel/Assign Discriminator/prob/kernel/read:026Discriminator/prob/kernel/Initializer/random_uniform:08

Discriminator/prob/bias:0Discriminator/prob/bias/AssignDiscriminator/prob/bias/read:02+Discriminator/prob/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
И
EDiscriminator/first_layer/fully_connected/kernel/discriminator_opti:0JDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/AssignJDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/read:02WDiscriminator/first_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros:0
Р
GDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1:0LDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/AssignLDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/read:02YDiscriminator/first_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros:0
А
CDiscriminator/first_layer/fully_connected/bias/discriminator_opti:0HDiscriminator/first_layer/fully_connected/bias/discriminator_opti/AssignHDiscriminator/first_layer/fully_connected/bias/discriminator_opti/read:02UDiscriminator/first_layer/fully_connected/bias/discriminator_opti/Initializer/zeros:0
И
EDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1:0JDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/AssignJDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/read:02WDiscriminator/first_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros:0
М
FDiscriminator/second_layer/fully_connected/kernel/discriminator_opti:0KDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/AssignKDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/read:02XDiscriminator/second_layer/fully_connected/kernel/discriminator_opti/Initializer/zeros:0
Ф
HDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1:0MDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/AssignMDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/read:02ZDiscriminator/second_layer/fully_connected/kernel/discriminator_opti_1/Initializer/zeros:0
Д
DDiscriminator/second_layer/fully_connected/bias/discriminator_opti:0IDiscriminator/second_layer/fully_connected/bias/discriminator_opti/AssignIDiscriminator/second_layer/fully_connected/bias/discriminator_opti/read:02VDiscriminator/second_layer/fully_connected/bias/discriminator_opti/Initializer/zeros:0
М
FDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1:0KDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/AssignKDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/read:02XDiscriminator/second_layer/fully_connected/bias/discriminator_opti_1/Initializer/zeros:0
м
.Discriminator/prob/kernel/discriminator_opti:03Discriminator/prob/kernel/discriminator_opti/Assign3Discriminator/prob/kernel/discriminator_opti/read:02@Discriminator/prob/kernel/discriminator_opti/Initializer/zeros:0
ф
0Discriminator/prob/kernel/discriminator_opti_1:05Discriminator/prob/kernel/discriminator_opti_1/Assign5Discriminator/prob/kernel/discriminator_opti_1/read:02BDiscriminator/prob/kernel/discriminator_opti_1/Initializer/zeros:0
д
,Discriminator/prob/bias/discriminator_opti:01Discriminator/prob/bias/discriminator_opti/Assign1Discriminator/prob/bias/discriminator_opti/read:02>Discriminator/prob/bias/discriminator_opti/Initializer/zeros:0
м
.Discriminator/prob/bias/discriminator_opti_1:03Discriminator/prob/bias/discriminator_opti_1/Assign3Discriminator/prob/bias/discriminator_opti_1/read:02@Discriminator/prob/bias/discriminator_opti_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0

;Encoder/first_layer/fully_connected/kernel/generator_opti:0@Encoder/first_layer/fully_connected/kernel/generator_opti/Assign@Encoder/first_layer/fully_connected/kernel/generator_opti/read:02MEncoder/first_layer/fully_connected/kernel/generator_opti/Initializer/zeros:0

=Encoder/first_layer/fully_connected/kernel/generator_opti_1:0BEncoder/first_layer/fully_connected/kernel/generator_opti_1/AssignBEncoder/first_layer/fully_connected/kernel/generator_opti_1/read:02OEncoder/first_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros:0

9Encoder/first_layer/fully_connected/bias/generator_opti:0>Encoder/first_layer/fully_connected/bias/generator_opti/Assign>Encoder/first_layer/fully_connected/bias/generator_opti/read:02KEncoder/first_layer/fully_connected/bias/generator_opti/Initializer/zeros:0

;Encoder/first_layer/fully_connected/bias/generator_opti_1:0@Encoder/first_layer/fully_connected/bias/generator_opti_1/Assign@Encoder/first_layer/fully_connected/bias/generator_opti_1/read:02MEncoder/first_layer/fully_connected/bias/generator_opti_1/Initializer/zeros:0

<Encoder/second_layer/fully_connected/kernel/generator_opti:0AEncoder/second_layer/fully_connected/kernel/generator_opti/AssignAEncoder/second_layer/fully_connected/kernel/generator_opti/read:02NEncoder/second_layer/fully_connected/kernel/generator_opti/Initializer/zeros:0

>Encoder/second_layer/fully_connected/kernel/generator_opti_1:0CEncoder/second_layer/fully_connected/kernel/generator_opti_1/AssignCEncoder/second_layer/fully_connected/kernel/generator_opti_1/read:02PEncoder/second_layer/fully_connected/kernel/generator_opti_1/Initializer/zeros:0

:Encoder/second_layer/fully_connected/bias/generator_opti:0?Encoder/second_layer/fully_connected/bias/generator_opti/Assign?Encoder/second_layer/fully_connected/bias/generator_opti/read:02LEncoder/second_layer/fully_connected/bias/generator_opti/Initializer/zeros:0

<Encoder/second_layer/fully_connected/bias/generator_opti_1:0AEncoder/second_layer/fully_connected/bias/generator_opti_1/AssignAEncoder/second_layer/fully_connected/bias/generator_opti_1/read:02NEncoder/second_layer/fully_connected/bias/generator_opti_1/Initializer/zeros:0
 
?Encoder/second_layer/batch_normalization/gamma/generator_opti:0DEncoder/second_layer/batch_normalization/gamma/generator_opti/AssignDEncoder/second_layer/batch_normalization/gamma/generator_opti/read:02QEncoder/second_layer/batch_normalization/gamma/generator_opti/Initializer/zeros:0
Ј
AEncoder/second_layer/batch_normalization/gamma/generator_opti_1:0FEncoder/second_layer/batch_normalization/gamma/generator_opti_1/AssignFEncoder/second_layer/batch_normalization/gamma/generator_opti_1/read:02SEncoder/second_layer/batch_normalization/gamma/generator_opti_1/Initializer/zeros:0

>Encoder/second_layer/batch_normalization/beta/generator_opti:0CEncoder/second_layer/batch_normalization/beta/generator_opti/AssignCEncoder/second_layer/batch_normalization/beta/generator_opti/read:02PEncoder/second_layer/batch_normalization/beta/generator_opti/Initializer/zeros:0
Є
@Encoder/second_layer/batch_normalization/beta/generator_opti_1:0EEncoder/second_layer/batch_normalization/beta/generator_opti_1/AssignEEncoder/second_layer/batch_normalization/beta/generator_opti_1/read:02REncoder/second_layer/batch_normalization/beta/generator_opti_1/Initializer/zeros:0
Ь
*Encoder/encoder_mu/kernel/generator_opti:0/Encoder/encoder_mu/kernel/generator_opti/Assign/Encoder/encoder_mu/kernel/generator_opti/read:02<Encoder/encoder_mu/kernel/generator_opti/Initializer/zeros:0
д
,Encoder/encoder_mu/kernel/generator_opti_1:01Encoder/encoder_mu/kernel/generator_opti_1/Assign1Encoder/encoder_mu/kernel/generator_opti_1/read:02>Encoder/encoder_mu/kernel/generator_opti_1/Initializer/zeros:0
Ф
(Encoder/encoder_mu/bias/generator_opti:0-Encoder/encoder_mu/bias/generator_opti/Assign-Encoder/encoder_mu/bias/generator_opti/read:02:Encoder/encoder_mu/bias/generator_opti/Initializer/zeros:0
Ь
*Encoder/encoder_mu/bias/generator_opti_1:0/Encoder/encoder_mu/bias/generator_opti_1/Assign/Encoder/encoder_mu/bias/generator_opti_1/read:02<Encoder/encoder_mu/bias/generator_opti_1/Initializer/zeros:0
м
.Encoder/encoder_logvar/kernel/generator_opti:03Encoder/encoder_logvar/kernel/generator_opti/Assign3Encoder/encoder_logvar/kernel/generator_opti/read:02@Encoder/encoder_logvar/kernel/generator_opti/Initializer/zeros:0
ф
0Encoder/encoder_logvar/kernel/generator_opti_1:05Encoder/encoder_logvar/kernel/generator_opti_1/Assign5Encoder/encoder_logvar/kernel/generator_opti_1/read:02BEncoder/encoder_logvar/kernel/generator_opti_1/Initializer/zeros:0
д
,Encoder/encoder_logvar/bias/generator_opti:01Encoder/encoder_logvar/bias/generator_opti/Assign1Encoder/encoder_logvar/bias/generator_opti/read:02>Encoder/encoder_logvar/bias/generator_opti/Initializer/zeros:0
м
.Encoder/encoder_logvar/bias/generator_opti_1:03Encoder/encoder_logvar/bias/generator_opti_1/Assign3Encoder/encoder_logvar/bias/generator_opti_1/read:02@Encoder/encoder_logvar/bias/generator_opti_1/Initializer/zeros:0TЙCќ       Ъ{­	k.џжA*№
u
generator_loss_1*a	   ЅPю?   ЅPю?      №?!   ЅPю?) ђзИь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    Гє?    Гє?      №?!    Гє?)@МѓXoЧњ?2КP1ѓ?3?шЏ|ѕ?џџџџџџя:              №?         Эпдў       л 	LШО.џжA(*№
u
generator_loss_1*a	   РїЙэ?   РїЙэ?      №?!   РїЙэ?) ЬIы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    КЭГ?    КЭГ?      №?!    КЭГ?)@ЮФ№x?2І{ ЈЧГГ? l(ЌЕ?џџџџџџя:              №?        2}йМў       л 	Хою.џжAP*№
u
generator_loss_1*a	    Iэ?    Iэ?      №?!    Iэ?) eOVЮkы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   !?   !?      №?!   !?) DWWKT9?2RcУн?^ЇSНР?џџџџџџя:              №?        лў       л 	|"/џжAx*№
u
generator_loss_1*a	   `Иэ?   `Иэ?      №?!   `Иэ?) GvЗы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @иf?   @иf?      №?!   @иf?)`Mlќ)$?2eiSЮm?#Ї+(Х?џџџџџџя:              №?        Э џ       SтІ	ЂZ/џжA *№
u
generator_loss_1*a	   `БЖэ?   `БЖэ?      №?!   `БЖэ?) /{4ы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    шz?    шz?      №?!    шz?) jњТ?2oЯ5sz?ЫДЪT}?џџџџџџя:              №?        Џџ       SтІ	S/џжAШ*№
u
generator_loss_1*a	   `a­э?   `a­э?      №?!   `a­э?) Oфыы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   рњrx?   рњrx?      №?!   рњrx?) в*sЎ?2*QHЅx?oЯ5sz?џџџџџџя:              №?        Ћd*џ       SтІ	mўг/џжA№*№
u
generator_loss_1*a	   РIЕэ?   РIЕэ?      №?!   РIЕэ?)јѕЖы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   Р\_d?   Р\_d?      №?!   Р\_d?) Љ? №й>2ЊЮ%b?5Ucv0ed?џџџџџџя:              №?        ЗЉѓџ       SтІ	Э0џжA*№
u
generator_loss_1*a	   @oМэ?   @oМэ?      №?!   @oМэ?)ФВ@пЁы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   %i?   %i?      №?!   %i?) ђcЉу>2P}Ѓ­h?пЄж(g%k?џџџџџџя:              №?        ыч;Тџ       SтІ	Ќ]0џжAР*№
u
generator_loss_1*a	    зФэ?    зФэ?      №?!    зФэ?) ёЉБы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @жg?   @жg?      №?!   @жg?)xююmс>2Tw шNof?P}Ѓ­h?џџџџџџя:              №?        ІЦ}џ       SтІ	i|Ї0џжAш*№
u
generator_loss_1*a	   @Зэ?   @Зэ?      №?!   @Зэ?)аJойы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   ЁR?   ЁR?      №?!   ЁR?) !|КАЕ>2nKLQ?lDZrS?џџџџџџя:              №?         иѓфџ       SтІ	Эhќ0џжA*№
u
generator_loss_1*a	   Дэ?   Дэ?      №?!   Дэ?) ЂФ|*ы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @ЫX?   @ЫX?      №?!   @ЫX?)єFрЪТ>2мSsW?ІРbBхSY?џџџџџџя:              №?        l5џ       SтІ	№ЇQ1џжAИ*№
u
generator_loss_1*a	   РИЈэ?   РИЈэ?      №?!   РИЈэ?)Є
tH}ы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   ыk?   ыk?      №?!   ыk?) ЂџУ}[ш>2пЄж(g%k?іNрWмm?џџџџџџя:              №?        П
0џ       SтІ	AЂЈ1џжAр*№
u
generator_loss_1*a	   @аКэ?   @аКэ?      №?!   @аКэ?)@3мы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    нT?    нT?      №?!    нT?)@Lджс5Л>2lDZrS?<DKcюT?џџџџџџя:              №?        кжчBџ       SтІ	K
2џжA*№
u
generator_loss_1*a	   Рэ?   Рэ?      №?!   Рэ?)ЬVakы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @|Q?   @|Q?      №?!   @|Q?) сD3В>2k 1^њsO?nKLQ?џџџџџџя:              №?        !9\\џ       SтІ	HSj2џжAА*№
u
generator_loss_1*a	   @|Тэ?   @|Тэ?      №?!   @|Тэ?)pj8­ы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    fU?    fU?      №?!    fU?)@>rЙ%Н>2<DKcюT?мSsW?џџџџџџя:              №?        №љџ       SтІ	мZа2џжAи*№
u
generator_loss_1*a	   @Мэ?   @Мэ?      №?!   @Мэ?)NJЁы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   х`?   х`?      №?!   х`?) ФhHузб>2lъPл`?ЊЮ%b?џџџџџџя:              №?        ьЫН7џ       SтІ	ІЅ=3џжA*№
u
generator_loss_1*a	   Ыэ?   Ыэ?      №?!   Ыэ?) bІьНы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @ќV?   @ќV?      №?!   @ќV?) сЈlО>2<DKcюT?мSsW?џџџџџџя:              №?        Cd`џ       SтІ	ћЗЉ3џжAЈ*№
u
generator_loss_1*a	    Йэ?    Йэ?      №?!    Йэ?)  0	oы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    НA?    НA?      №?!    НA?) јaM>2Ь№#@?сД!СA?џџџџџџя:              №?        _ЬСњџ       SтІ	y2&4џжAа*№
u
generator_loss_1*a	    oЕэ?    oЕэ?      №?!    oЕэ?) цоы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   ``C?   ``C?      №?!   ``C?)@"Ѕ{v>2сД!СA?TФЅC?џџџџџџя:              №?        ыџ       SтІ	пў4џжAј*№
u
generator_loss_1*a	    Ээ?    Ээ?      №?!    Ээ?) L:шШСы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   SA?   SA?      №?!   SA?) Є-W8У>2Ь№#@?сД!СA?џџџџџџя:              №?        Ђџ       SтІ	Ю5џжA *№
u
generator_loss_1*a	   Qлэ?   Qлэ?      №?!   Qлэ?) Wмbлы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    Хц3?    Хц3?      №?!    Хц3?)@њ@зСx>2 О82?ъuw74?џџџџџџя:              №?        м8Mџ       SтІ	5џжAШ*№
u
generator_loss_1*a	    Фэ?    Фэ?      №?!    Фэ?) Ѓ5§Аы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    DiB?    DiB?      №?!    DiB?)  Ё/>2сД!СA?TФЅC?џџџџџџя:              №?        с!ќ       Ъ{­	m6џжA*№
u
generator_loss_1*a	   @№Їэ?   @№Їэ?      №?!   @№Їэ?)РЇЮд{ы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    5 6?    5 6?      №?!    5 6?)@К3Oд~>2Е%V6?uмЌХ@8?џџџџџџя:              №?        ЂMлШў       л 	/6џжA(*№
u
generator_loss_1*a	   р(Чэ?   р(Чэ?      №?!   р(Чэ?) 6бЕы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    ЇІH?    ЇІH?      №?!    ЇІH?) h^§Ђ>2
ТћЁG?qUћўI?џџџџџџя:              №?        3ёў       л 	($-7џжAP*№
u
generator_loss_1*a	   dЙэ?   dЙэ?      №?!   dЙэ?) ЂC;8ы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    Xж-?    Xж-?      №?!    Xж-?)  ђ9вk>2Ћ7Kaa+?ИеVlQ.?џџџџџџя:              №?        3&Їў       л 	Ъл7џжAx*№
u
generator_loss_1*a	   `'Шэ?   `'Шэ?      №?!   `'Шэ?) s`JЋЗы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   `Ђ<?   `Ђ<?      №?!   `Ђ<?) э%їс>2§Г%>І­:?dЌ\DX=?џџџџџџя:              №?        иеNџ       SтІ	Цџn8џжA *№
u
generator_loss_1*a	   Рaвэ?   Рaвэ?      №?!   Рaвэ?)"PИЪы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   nr@?   nr@?      №?!   nr@?) $iш>2Ь№#@?сД!СA?џџџџџџя:              №?         [xлџ       SтІ	a9џжAШ*№
u
generator_loss_1*a	    ѕпэ?    ѕпэ?      №?!    ѕпэ?) ВУБфы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @%;?   @%;?      №?!   @%;?)\gPZш>2§Г%>І­:?dЌ\DX=?џџџџџџя:              №?        ЬЃгџ       SтІ	ђ`Ы9џжA№*№
u
generator_loss_1*a	    РУэ?    РУэ?      №?!    РУэ?) %zЏы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   `ЛТD?   `ЛТD?      №?!   `ЛТD?)@Vjx№>2TФЅC?aУ$ќ{E?џџџџџџя:              №?        RЗ№џ       SтІ	[nz:џжA*№
u
generator_loss_1*a	    QЧэ?    QЧэ?      №?!    QЧэ?) 5rЖы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   Рlє=?   Рlє=?      №?!   Рlє=?)сP
>2dЌ\DX=?Ь№#@?џџџџџџя:              №?        rP3џ       SтІ	С-,;џжAР*№
u
generator_loss_1*a	   роэ?   роэ?      №?!   роэ?) Д	^сы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    A`>?    A`>?      №?!    A`>?) Fе>2dЌ\DX=?Ь№#@?џџџџџџя:              №?        aЉ
џ       SтІ	ЭхЬ;џжAш*№
u
generator_loss_1*a	    vЫэ?    vЫэ?      №?!    vЫэ?) ПХЌдНы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    %Љ<?    %Љ<?      №?!    %Љ<?) MЋ>2§Г%>І­:?dЌ\DX=?џџџџџџя:              №?        YДЙvџ       SтІ	юzq<џжA*№
u
generator_loss_1*a	   рѕдэ?   рѕдэ?      №?!   рѕдэ?) 49чЯы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    :ѕ4?    :ѕ4?      №?!    :ѕ4?)@(чдПs{>2ъuw74?Е%V6?џџџџџџя:              №?        ЯыqЛџ       SтІ	cP=џжAИ*№
u
generator_loss_1*a	   ОЭэ?   ОЭэ?      №?!   ОЭэ?) 6Ты?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   р[к3?   р[к3?      №?!   р[к3?)@Ї=>Ђx>2 О82?ъuw74?џџџџџџя:              №?        Џ[yџ       SтІ	ёТ=џжAр*№
u
generator_loss_1*a	   ммэ?   ммэ?      №?!   ммэ?) bЯDоы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   ??   ??      №?!   ??) %AЛ>2dЌ\DX=?Ь№#@?џџџџџџя:              №?        SЩђџ       SтІ	Ъs>џжA*№
u
generator_loss_1*a	   рvщэ?   рvщэ?      №?!   рvщэ?) зТЮѕы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   рyь7?   рyь7?      №?!   рyь7?) ,xЙТт>2Е%V6?uмЌХ@8?џџџџџџя:              №?        <џ       SтІ	kQ:?џжAА*№
u
generator_loss_1*a	   мэ?   мэ?      №?!   мэ?) "ЄНны?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   ` ,?   ` ,?      №?!   ` ,?) Bsui>2Ћ7Kaa+?ИеVlQ.?џџџџџџя:              №?        E%№џ       SтІ	/@џжAи*№
u
generator_loss_1*a	   РОэ?   РОэ?      №?!   РОэ?)РYгЅы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   Р7Б(?   Р7Б(?      №?!   Р7Б(?) Љc>2+A Fр &?IсIч)ф(?џџџџџџя:              №?        QЕ9lџ       SтІ	­и@џжA*№
u
generator_loss_1*a	   зЮэ?   зЮэ?      №?!   зЮэ?) B;Фы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    Eн9?    Eн9?      №?!    Eн9?) ШЄбЕч>2uмЌХ@8?§Г%>І­:?џџџџџџя:              №?        ЬЈ|ўџ       SтІ	іЊAџжAЈ*№
u
generator_loss_1*a	    lкэ?    lкэ?      №?!    lкэ?) ЛtЫЗйы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    Сi3?    Сi3?      №?!    Сi3?)@*{тцw>2 О82?ъuw74?џџџџџџя:              №?        сmЃџ       SтІ	>BџжAа*№
u
generator_loss_1*a	   рKёэ?   рKёэ?      №?!   рKёэ?) шuь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @*-?   @*-?      №?!   @*-?)Cj>2Ћ7Kaa+?ИеVlQ.?џџџџџџя:              №?        ѕъtџ       SтІ	-RCџжAј*№
u
generator_loss_1*a	    rхэ?    rхэ?      №?!    rхэ?)  6ЩKюы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    6?    6?      №?!    6?)@0ЎЌ>2Е%V6?uмЌХ@8?џџџџџџя:              №?        ЌSшџ       SтІ	цg)DџжA *№
u
generator_loss_1*a	    cшэ?    cшэ?      №?!    cшэ?) )Ц7Ьѓы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   рЌF3?   рЌF3?      №?!   рЌF3?)@мѓУ9w>2 О82?ъuw74?џџџџџџя:              №?        3џ       SтІ	нўDџжAШ*№
u
generator_loss_1*a	   ртэ?   ртэ?      №?!   ртэ?) ъ"шїчы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    Q-?    Q-?      №?!    Q-?) L­Слj>2Ћ7Kaa+?ИеVlQ.?џџџџџџя:              №?        {ь­шќ       Ъ{­	чЕШEџжA*№
u
generator_loss_1*a	    nээ?    nээ?      №?!    nээ?) o 2:§ы?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   Рл`?   Рл`?      №?!   Рл`?) !(и-а>2EХи{Ѕ^?lъPл`?џџџџџџя:              №?        юо
иў       л 	шFџжA(*№
u
generator_loss_1*a	   рНїэ?   рНїэ?      №?!   рНїэ?) Є%ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    UБ]?    UБ]?      №?!    UБ]?) юAЫ>2m9ќHм[?EХи{Ѕ^?џџџџџџя:              №?        ыESў       л 	МGџжAP*№
u
generator_loss_1*a	   @Їю?   @Їю?      №?!   @Їю?)$Цк&ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @4<?   @4<?      №?!   @4<?)м:c?м>2§Г%>І­:?dЌ\DX=?џџџџџџя:              №?        ќў       л 	u^HџжAx*№
u
generator_loss_1*a	   `Щђэ?   `Щђэ?      №?!   `Щђэ?) ?П?ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    0ЦJ?    0ЦJ?      №?!    0ЦJ?)  HrйfІ>2qUћўI?IcDсњL?џџџџџџя:              №?        тQ/џ       SтІ	9BIџжA *№
u
generator_loss_1*a	    к№э?    к№э?      №?!    к№э?)  Эыь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   "r?   "r?      №?!   "r?) dљfє>2uWyтгr?hyOпs?џџџџџџя:              №?        Ћ)Жџ       SтІ	:И&JџжAШ*№
u
generator_loss_1*a	   рШю?   рШю?      №?!   рШю?) і"о9ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    +?    +?      №?!    +?)  IqЂ4?2ъ Б&?RcУн?џџџџџџя:              №?        Q^Ќџ       SтІ	Т%KџжA№*№
u
generator_loss_1*a	   `ю?   `ю?      №?!   `ю?) НвмRLь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @%KG?   @%KG?      №?!   @%KG?)\ЧЖє >2aУ$ќ{E?
ТћЁG?џџџџџџя:              №?        Тџ       SтІ	§ЙLџжA*№
u
generator_loss_1*a	   р}ю?   р}ю?      №?!   р}ю?) $%sZHь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   PK?   PK?      №?!   PK?) ZзСЇ>2qUћўI?IcDсњL?џџџџџџя:              №?        кЂ%Aџ       SтІ	x=MџжAР*№
u
generator_loss_1*a	   @Nю?   @Nю?      №?!   @Nю?)XwkяXь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   Ъ_?   Ъ_?      №?!   Ъ_?) 2Єi^Я>2EХи{Ѕ^?lъPл`?џџџџџџя:              №?        t{џ       SтІ	~6NџжAш*№
u
generator_loss_1*a	    V.ю?    V.ю?      №?!    V.ю?) і%wь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   РУСЄ?   РУСЄ?      №?!   РУСЄ?) свыэZ?2`Юлa8Є?б/и*>І?џџџџџџя:              №?        эЕ0аџ       SтІ	IЮYOџжA*№
u
generator_loss_1*a	   РЇю?   РЇю?      №?!   РЇю?)`МRь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   `8ГC?   `8ГC?      №?!   `8ГC?)@Ђ*b}A>2TФЅC?aУ$ќ{E?џџџџџџя:              №?        эdџ       SтІ	ЉtgPџжAИ*№
u
generator_loss_1*a	    С0ю?    С0ю?      №?!    С0ю?) эcД{ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   Р3F?   Р3F?      №?!   Р3F?) ЉGЮ>2aУ$ќ{E?
ТћЁG?џџџџџџя:              №?        Sџ       SтІ	ЃЪQџжAр*№
u
generator_loss_1*a	    ю?    ю?      №?!    ю?) бћЕSь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   `§M?   `§M?      №?!   `§M?) 7C2cЊ>2IcDсњL?k 1^њsO?џџџџџџя:              №?        |јрџ       SтІ	DRџжA*№
u
generator_loss_1*a	    м(ю?    м(ю?      №?!    м(ю?) йбlь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    Х?F?    Х?F?      №?!    Х?F?) йш[№>2aУ$ќ{E?
ТћЁG?џџџџџџя:              №?        Тceџ       SтІ	ё:SџжAА*№
u
generator_loss_1*a	   рз&ю?   рз&ю?      №?!   рз&ю?) Pdъiь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   РJ5?   РJ5?      №?!   РJ5?) 9Uин{>2ъuw74?Е%V6?џџџџџџя:              №?        ИНџ       SтІ	@ЗTџжAи*№
u
generator_loss_1*a	    Пю?    Пю?      №?!    Пю?) Ѓ=ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   ё?   ё?      №?!   ё?) $нn4>2МdЯr?У5еi}1?џџџџџџя:              №?        ЅбXDџ       SтІ	QгUџжA*№
u
generator_loss_1*a	   `Ш7ю?   `Ш7ю?      №?!   `Ш7ю?) БАёјь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    fR!?    fR!?      №?!    fR!?)@иS]СРR>2SF !?№[^:Г"?џџџџџџя:              №?        DGџ       SтІ	ьVџжAЈ*№
u
generator_loss_1*a	   @гю?   @гю?      №?!   @гю?)ЩщYь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   пк?   пк?      №?!   пк?) ICжуD>2щЋфк?Н.з?џџџџџџя:              №?        ЯgWџ       SтІ	ФXџжAа*№
u
generator_loss_1*a	   Р~*ю?   Р~*ю?      №?!   Р~*ю?)Ўцoь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   Ў4?   Ў4?      №?!   Ў4?) V Мz>2ъuw74?Е%V6?џџџџџџя:              №?        ЇіkЃџ       SтІ	Ъ.YџжAј*№
u
generator_loss_1*a	    Эю?    Эю?      №?!    Эю?) тBn?ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @`Э?   @`Э?      №?!   @`Э?)UтrF>2щЋфк?Н.з?џџџџџџя:              №?        CаЖ?џ       SтІ	Р=QZџжA *№
u
generator_loss_1*a	    §њэ?    §њэ?      №?!    §њэ?) Bцdь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    ыF:?    ыF:?      №?!    ыF:?) k?л>2uмЌХ@8?§Г%>І­:?џџџџџџя:              №?        (Аіџ       SтІ	С[џжAШ*№
u
generator_loss_1*a	   `uю?   `uю?      №?!   `uю?) \=*ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   `Рu6?   `Рu6?      №?!   `Рu6?)@3>2Е%V6?uмЌХ@8?џџџџџџя:              №?        ув1ќ       Ъ{­	\џжA*№
u
generator_loss_1*a	   @ю?   @ю?      №?!   @ю?)(аКEь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   РЖпK?   РЖпK?      №?!   РЖпK?)Ќ'c GЈ>2qUћўI?IcDсњL?џџџџџџя:              №?        5-яў       л 	"Hб]џжA(*№
u
generator_loss_1*a	   @Wю?   @Wю?      №?!   @Wю?)ф\ Yь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   `3a?   `3a?      №?!   `3a?)@.%2~в>2lъPл`?ЊЮ%b?џџџџџџя:              №?        Яў       л 	яќ§^џжAP*№
u
generator_loss_1*a	   `:8ю?   `:8ю?      №?!   `:8ю?) }К@аь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    5S?    5S?      №?!    5S?)@фHТЗ>2lDZrS?<DKcюT?џџџџџџя:              №?        {ў       л 	ЋЌ1`џжAx*№
u
generator_loss_1*a	    ,,ю?    ,,ю?      №?!    ,,ю?) иЕsь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    ЁK2?    ЁK2?      №?!    ЁK2?) ДЛыt>2 О82?ъuw74?џџџџџџя:              №?        П=8џ       SтІ	NgaџжA *№
u
generator_loss_1*a	   РC+ю?   РC+ю?      №?!   РC+ю?)pЃYqь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    -ќ??    -ќ??      №?!    -ќ??) йД[ј>2dЌ\DX=?Ь№#@?џџџџџџя:              №?        Rчпџ       SтІ	ёCЁbџжAШ*№
u
generator_loss_1*a	   `8ю?   `8ю?      №?!   `8ю?) %Ьь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   `v?B?   `v?B?      №?!   `v?B?)@ЪЦЯ>2сД!СA?TФЅC?џџџџџџя:              №?        lайџџ       SтІ	Щ(чcџжA№*№
u
generator_loss_1*a	   ЕQю?   ЕQю?      №?!   ЕQю?) rнђКь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   р|J3?   р|J3?      №?!   р|J3?)@цZCBw>2 О82?ъuw74?џџџџџџя:              №?        ­шDџ       SтІ	Рщ1eџжA*№
u
generator_loss_1*a	   @)4ю?   @)4ю?      №?!   @)4ю?),E^"ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    ТЪ@?    ТЪ@?      №?!    ТЪ@?) @Аk>2Ь№#@?сД!СA?џџџџџџя:              №?        FИєџ       SтІ	,fџжAР*№
u
generator_loss_1*a	   Рщ(ю?   Рщ(ю?      №?!   Рщ(ю?)xъlь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @xИH?   @xИH?      №?!   @xИH?)рЃШлЃ>2
ТћЁG?qUћўI?џџџџџџя:              №?        т9Оџ       SтІ	и^сgџжAш*№
u
generator_loss_1*a	   р0Mю?   р0Mю?      №?!   р0Mю?) ІзuБь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    kђ>?    kђ>?      №?!    kђ>?) љ=IЖэ>2dЌ\DX=?Ь№#@?џџџџџџя:              №?        VXћ]џ       SтІ	_џ.iџжA*№
u
generator_loss_1*a	    ќ0ю?    ќ0ю?      №?!    ќ0ю?) x"И#|ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    iе*?    iе*?      №?!    iе*?) (пYf>2IсIч)ф(?Ћ7Kaa+?џџџџџџя:              №?        Ё|Cџ       SтІ	фЂjџжAИ*№
u
generator_loss_1*a	     Tю?     Tю?      №?!     Tю?)  ЈМ\Оь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   `н?   `н?      №?!   `н?) wеGе&>2fъЪѕ7
?>hЦ'з?џџџџџџя:              №?        ўђФџ       SтІ	lџжAр*№
u
generator_loss_1*a	   ш$ю?   ш$ю?      №?!   ш$ю?) BЙ^eь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   п?   п?      №?!   п?) 2/.Y@>2НT7Сж?аvVЁR9?џџџџџџя:              №?        nМ-Tџ       SтІ	lmџжA*№
u
generator_loss_1*a	    Ею?    Ею?      №?!    Ею?) нІE]?ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   рLо#?   рLо#?      №?!   рLо#?)@\Й*ЌX>2№[^:Г"?U4@@$?џџџџџџя:              №?        =fџ       SтІ	n
oџжAА*№
u
generator_loss_1*a	    в8ю?    в8ю?      №?!    в8ю?)  bЄюь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @8с?   @8с?      №?!   @8с?) СЭЦљњ3>2МdЯr?У5еi}1?џџџџџџя:              №?        ?Ўџ       SтІ	@ИipџжAи*№
u
generator_loss_1*a	   рp'ю?   рp'ю?      №?!   рp'ю?) &А@$jь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    2Ы[?    2Ы[?      №?!    2Ы[?) Dкю#Ш>2ІРbBхSY?m9ќHм[?џџџџџџя:              №?        kJџ       SтІ	З$ЪqџжA*№
u
generator_loss_1*a	   р
8ю?   р
8ю?      №?!   р
8ю?) Вvь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   рЊO?   рЊO?      №?!   рЊO?) r kЭЎ>2IcDсњL?k 1^њsO?џџџџџџя:              №?        юb6џ       SтІ	HE/sџжAЈ*№
u
generator_loss_1*a	   rю?   rю?      №?!   rю?) ВIЈ3Yь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   рх??   рх??      №?!   рх??) TQ	B>2dЌ\DX=?Ь№#@?џџџџџџя:              №?        мнџ       SтІ	ЕЄtџжAа*№
u
generator_loss_1*a	   e6ю?   e6ю?      №?!   e6ю?) ђбЧZь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   Рќ 5?   Рќ 5?      №?!   Рќ 5?) Љ{>2ъuw74?Е%V6?џџџџџџя:              №?        xіеџ       SтІ	"г vџжAј*№
u
generator_loss_1*a	   @ю?   @ю?      №?!   @ю?)аbOWь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   рІ7?   рІ7?      №?!   рІ7?) :$ЉСg>2Е%V6?uмЌХ@8?џџџџџџя:              №?        ђ2џ       SтІ	ЋGwџжA *№
u
generator_loss_1*a	   oю?   oю?      №?!   oю?) ќК@ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @вЫB?   @вЫB?      №?!   @вЫB?) бтН>2сД!СA?TФЅC?џџџџџџя:              №?        1бџ       SтІ	ЇHyџжAШ*№
u
generator_loss_1*a	    -ю?    -ю?      №?!    -ю?) 'ј~бtь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @&i6?   @&i6?      №?!   @&i6?) qc6мc>2Е%V6?uмЌХ@8?џџџџџџя:              №?        ЗГнАќ       Ъ{­	ЫІzџжA*№
u
generator_loss_1*a	   Ц&ю?   Ц&ю?      №?!   Ц&ю?) R?,уhь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   рт/C?   рт/C?      №?!   рт/C?)@u%J>2сД!СA?TФЅC?џџџџџџя:              №?        _ў       л 	Х5|џжA(*№
u
generator_loss_1*a	   |8ю?   |8ю?      №?!   |8ю?) bЄ%Mь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   @юC?   @юC?      №?!   @юC?) Бћ\З>2сД!СA?TФЅC?џџџџџџя:              №?        Ї­l`ў       л 	HuН}џжAP*№
u
generator_loss_1*a	   Яbю?   Яbю?      №?!   Яbю?) 1,vкь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    йPЛ?    йPЛ?      №?!    йPЛ?) ш;Q?2%gіcE9К?ЉЄ(!иМ?џџџџџџя:              №?        Нbў       л 	Щ>џжAx*№
u
generator_loss_1*a	   рE1ю?   рE1ю?      №?!   рE1ю?) тЎ|ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   8@?   8@?      №?!   8@?) ТcZюN>2ji69щ?SF !?џџџџџџя:              №?        \јє"џ       SтІ	ЕЦџжA *№
u
generator_loss_1*a	   `=ю?   `=ю?      №?!   `=ю?) uG]сь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   бЧ"?   бЧ"?      №?!   бЧ"?) $жVV>2№[^:Г"?U4@@$?џџџџџџя:              №?        &бF7џ       SтІ	ѓ UџжAШ*№
u
generator_loss_1*a	   2ю?   2ю?      №?!   2ю?) 2KЬE~ь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   рBГ?   рBГ?      №?!   рBГ?) ТЕ	t->2>hЦ'з?x?xЙ?џџџџџџя:              №?        ЋИїџ       SтІ	й=сџжA№*№
u
generator_loss_1*a	   рЁ&ю?   рЁ&ю?      №?!   рЁ&ю?) мІ'hь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   `EО*?   `EО*?      №?!   `EО*?) gjYf>2IсIч)ф(?Ћ7Kaa+?џџџџџџя:              №?        ЇЅџ       SтІ	уFsџжA*№
u
generator_loss_1*a	   р;ю?   р;ю?      №?!   р;ю?) цЩь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   жb?   жb?      №?!   жb?) ЄлШ0>2x?xЙ?МdЯr?џџџџџџя:              №?        Ф lџ       SтІ	ЗџжAР*№
u
generator_loss_1*a	   РIю?   РIю?      №?!   РIю?)иkЊь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    i?    i?      №?!    i?) ШЛѕc*>2>hЦ'з?x?xЙ?џџџџџџя:              №?        GЁTџ       SтІ	Ё^џжAш*№
u
generator_loss_1*a	   @В[ю?   @В[ю?      №?!   @В[ю?)шЌљєЬь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	   Р*	?   Р*	?      №?!   Р*	?)}Ъ$>2ЂчШе?fъЪѕ7
?џџџџџџя:              №?        Њщ!џ       SтІ	 зaџжA*№
u
generator_loss_1*a	    4ю?    4ю?      №?!    4ю?) ш:BDь?2iZэ?+њЭО$№?џџџџџџя:              №?        
w
discriminator_loss*a	    ыт;?    ыт;?      №?!    ыт;?) ШЎ5M>2§Г%>І­:?dЌ\DX=?џџџџџџя:              №?        -юњ