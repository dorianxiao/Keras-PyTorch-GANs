       £K"	  @dкэ÷Abrain.Event:28 4Г	     дg€	уИddкэ÷A"цТ
u
Generator/noise_inPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€d*
shape:€€€€€€€€€d
я
MGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   А   
—
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&Њ
—
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&>
∆
UGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	dА*

seed *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
seed2 
ќ
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
: 
б
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	dА
”
GGenerator/first_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	dА
г
,Generator/first_layer/fully_connected/kernel
VariableV2*
shape:	dА*
dtype0*
_output_shapes
:	dА*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container 
»
3Generator/first_layer/fully_connected/kernel/AssignAssign,Generator/first_layer/fully_connected/kernelGGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	dА
÷
1Generator/first_layer/fully_connected/kernel/readIdentity,Generator/first_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	dА
 
<Generator/first_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
„
*Generator/first_layer/fully_connected/bias
VariableV2*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
≥
1Generator/first_layer/fully_connected/bias/AssignAssign*Generator/first_layer/fully_connected/bias<Generator/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
ћ
/Generator/first_layer/fully_connected/bias/readIdentity*Generator/first_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:А
÷
,Generator/first_layer/fully_connected/MatMulMatMulGenerator/noise_in1Generator/first_layer/fully_connected/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
б
-Generator/first_layer/fully_connected/BiasAddBiasAdd,Generator/first_layer/fully_connected/MatMul/Generator/first_layer/fully_connected/bias/read*(
_output_shapes
:€€€€€€€€€А*
T0*
data_formatNHWC
k
&Generator/first_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
µ
$Generator/first_layer/leaky_relu/mulMul&Generator/first_layer/leaky_relu/alpha-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
≥
 Generator/first_layer/leaky_reluMaximum$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
б
NGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"А      
”
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   Њ*
dtype0*
_output_shapes
: 
”
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
 
VGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformNGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
“
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
_output_shapes
: 
ж
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulVGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
Ў
HGenerator/second_layer/fully_connected/kernel/Initializer/random_uniformAddLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА*
T0
з
-Generator/second_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
АА
Ќ
4Generator/second_layer/fully_connected/kernel/AssignAssign-Generator/second_layer/fully_connected/kernelHGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
Џ
2Generator/second_layer/fully_connected/kernel/readIdentity-Generator/second_layer/fully_connected/kernel*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
ћ
=Generator/second_layer/fully_connected/bias/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ў
+Generator/second_layer/fully_connected/bias
VariableV2*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
Ј
2Generator/second_layer/fully_connected/bias/AssignAssign+Generator/second_layer/fully_connected/bias=Generator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
ѕ
0Generator/second_layer/fully_connected/bias/readIdentity+Generator/second_layer/fully_connected/bias*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:А
ж
-Generator/second_layer/fully_connected/MatMulMatMul Generator/first_layer/leaky_relu2Generator/second_layer/fully_connected/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
д
.Generator/second_layer/fully_connected/BiasAddBiasAdd-Generator/second_layer/fully_connected/MatMul0Generator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
’
AGenerator/second_layer/batch_normalization/gamma/Initializer/onesConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueBА*  А?*
dtype0*
_output_shapes	
:А
г
0Generator/second_layer/batch_normalization/gamma
VariableV2*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
 
7Generator/second_layer/batch_normalization/gamma/AssignAssign0Generator/second_layer/batch_normalization/gammaAGenerator/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
ё
5Generator/second_layer/batch_normalization/gamma/readIdentity0Generator/second_layer/batch_normalization/gamma*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:А
‘
AGenerator/second_layer/batch_normalization/beta/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
б
/Generator/second_layer/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:А
«
6Generator/second_layer/batch_normalization/beta/AssignAssign/Generator/second_layer/batch_normalization/betaAGenerator/second_layer/batch_normalization/beta/Initializer/zeros*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
џ
4Generator/second_layer/batch_normalization/beta/readIdentity/Generator/second_layer/batch_normalization/beta*
_output_shapes	
:А*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
в
HGenerator/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
valueBА*    *
dtype0*
_output_shapes	
:А
п
6Generator/second_layer/batch_normalization/moving_mean
VariableV2*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
г
=Generator/second_layer/batch_normalization/moving_mean/AssignAssign6Generator/second_layer/batch_normalization/moving_meanHGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:А
р
;Generator/second_layer/batch_normalization/moving_mean/readIdentity6Generator/second_layer/batch_normalization/moving_mean*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
_output_shapes	
:А
й
KGenerator/second_layer/batch_normalization/moving_variance/Initializer/onesConst*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
valueBА*  А?*
dtype0*
_output_shapes	
:А
ч
:Generator/second_layer/batch_normalization/moving_variance
VariableV2*
shared_name *M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
	container *
shape:А*
dtype0*
_output_shapes	
:А
т
AGenerator/second_layer/batch_normalization/moving_variance/AssignAssign:Generator/second_layer/batch_normalization/moving_varianceKGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:А*
use_locking(
ь
?Generator/second_layer/batch_normalization/moving_variance/readIdentity:Generator/second_layer/batch_normalization/moving_variance*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
_output_shapes	
:А

:Generator/second_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
в
8Generator/second_layer/batch_normalization/batchnorm/addAdd?Generator/second_layer/batch_normalization/moving_variance/read:Generator/second_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:А*
T0
£
:Generator/second_layer/batch_normalization/batchnorm/RsqrtRsqrt8Generator/second_layer/batch_normalization/batchnorm/add*
_output_shapes	
:А*
T0
Ў
8Generator/second_layer/batch_normalization/batchnorm/mulMul:Generator/second_layer/batch_normalization/batchnorm/Rsqrt5Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:А
ё
:Generator/second_layer/batch_normalization/batchnorm/mul_1Mul.Generator/second_layer/fully_connected/BiasAdd8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:€€€€€€€€€А
ё
:Generator/second_layer/batch_normalization/batchnorm/mul_2Mul;Generator/second_layer/batch_normalization/moving_mean/read8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
„
8Generator/second_layer/batch_normalization/batchnorm/subSub4Generator/second_layer/batch_normalization/beta/read:Generator/second_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:А
к
:Generator/second_layer/batch_normalization/batchnorm/add_1Add:Generator/second_layer/batch_normalization/batchnorm/mul_18Generator/second_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:€€€€€€€€€А
l
'Generator/second_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
ƒ
%Generator/second_layer/leaky_relu/mulMul'Generator/second_layer/leaky_relu/alpha:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
¬
!Generator/second_layer/leaky_reluMaximum%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:€€€€€€€€€А*
T0
я
MGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
—
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *уµљ*
dtype0*
_output_shapes
: 
—
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *уµ=*
dtype0*
_output_shapes
: 
«
UGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
seed2 
ќ
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
_output_shapes
: 
в
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АА*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
‘
GGenerator/third_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
е
,Generator/third_layer/fully_connected/kernel
VariableV2*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
…
3Generator/third_layer/fully_connected/kernel/AssignAssign,Generator/third_layer/fully_connected/kernelGGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
„
1Generator/third_layer/fully_connected/kernel/readIdentity,Generator/third_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
АА
 
<Generator/third_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
„
*Generator/third_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:А
≥
1Generator/third_layer/fully_connected/bias/AssignAssign*Generator/third_layer/fully_connected/bias<Generator/third_layer/fully_connected/bias/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(
ћ
/Generator/third_layer/fully_connected/bias/readIdentity*Generator/third_layer/fully_connected/bias*
_output_shapes	
:А*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
е
,Generator/third_layer/fully_connected/MatMulMatMul!Generator/second_layer/leaky_relu1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
б
-Generator/third_layer/fully_connected/BiasAddBiasAdd,Generator/third_layer/fully_connected/MatMul/Generator/third_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
”
@Generator/third_layer/batch_normalization/gamma/Initializer/onesConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueBА*  А?*
dtype0*
_output_shapes	
:А
б
/Generator/third_layer/batch_normalization/gamma
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:А
∆
6Generator/third_layer/batch_normalization/gamma/AssignAssign/Generator/third_layer/batch_normalization/gamma@Generator/third_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
џ
4Generator/third_layer/batch_normalization/gamma/readIdentity/Generator/third_layer/batch_normalization/gamma*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:А
“
@Generator/third_layer/batch_normalization/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueBА*    
я
.Generator/third_layer/batch_normalization/beta
VariableV2*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А
√
5Generator/third_layer/batch_normalization/beta/AssignAssign.Generator/third_layer/batch_normalization/beta@Generator/third_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
Ў
3Generator/third_layer/batch_normalization/beta/readIdentity.Generator/third_layer/batch_normalization/beta*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:А*
T0
а
GGenerator/third_layer/batch_normalization/moving_mean/Initializer/zerosConst*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
valueBА*    *
dtype0*
_output_shapes	
:А
н
5Generator/third_layer/batch_normalization/moving_mean
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
	container *
shape:А
я
<Generator/third_layer/batch_normalization/moving_mean/AssignAssign5Generator/third_layer/batch_normalization/moving_meanGGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:А
н
:Generator/third_layer/batch_normalization/moving_mean/readIdentity5Generator/third_layer/batch_normalization/moving_mean*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
_output_shapes	
:А
з
JGenerator/third_layer/batch_normalization/moving_variance/Initializer/onesConst*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
valueBА*  А?*
dtype0*
_output_shapes	
:А
х
9Generator/third_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
	container *
shape:А
о
@Generator/third_layer/batch_normalization/moving_variance/AssignAssign9Generator/third_layer/batch_normalization/moving_varianceJGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:А
щ
>Generator/third_layer/batch_normalization/moving_variance/readIdentity9Generator/third_layer/batch_normalization/moving_variance*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
_output_shapes	
:А
~
9Generator/third_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
я
7Generator/third_layer/batch_normalization/batchnorm/addAdd>Generator/third_layer/batch_normalization/moving_variance/read9Generator/third_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:А*
T0
°
9Generator/third_layer/batch_normalization/batchnorm/RsqrtRsqrt7Generator/third_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:А
’
7Generator/third_layer/batch_normalization/batchnorm/mulMul9Generator/third_layer/batch_normalization/batchnorm/Rsqrt4Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:А
џ
9Generator/third_layer/batch_normalization/batchnorm/mul_1Mul-Generator/third_layer/fully_connected/BiasAdd7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:€€€€€€€€€А
џ
9Generator/third_layer/batch_normalization/batchnorm/mul_2Mul:Generator/third_layer/batch_normalization/moving_mean/read7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
‘
7Generator/third_layer/batch_normalization/batchnorm/subSub3Generator/third_layer/batch_normalization/beta/read9Generator/third_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:А
з
9Generator/third_layer/batch_normalization/batchnorm/add_1Add9Generator/third_layer/batch_normalization/batchnorm/mul_17Generator/third_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:€€€€€€€€€А
k
&Generator/third_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
Ѕ
$Generator/third_layer/leaky_relu/mulMul&Generator/third_layer/leaky_relu/alpha9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
њ
 Generator/third_layer/leaky_reluMaximum$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
Ё
LGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ѕ
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  Аљ*
dtype0*
_output_shapes
: 
ѕ
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  А=
ƒ
TGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
seed2 
 
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/subSubJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
_output_shapes
: 
ё
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
АА
–
FGenerator/last_layer/fully_connected/kernel/Initializer/random_uniformAddJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
г
+Generator/last_layer/fully_connected/kernel
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
≈
2Generator/last_layer/fully_connected/kernel/AssignAssign+Generator/last_layer/fully_connected/kernelFGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
‘
0Generator/last_layer/fully_connected/kernel/readIdentity+Generator/last_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
АА
‘
KGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:А*
dtype0
ƒ
AGenerator/last_layer/fully_connected/bias/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
…
;Generator/last_layer/fully_connected/bias/Initializer/zerosFillKGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorAGenerator/last_layer/fully_connected/bias/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:А
’
)Generator/last_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:А
ѓ
0Generator/last_layer/fully_connected/bias/AssignAssign)Generator/last_layer/fully_connected/bias;Generator/last_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
…
.Generator/last_layer/fully_connected/bias/readIdentity)Generator/last_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:А
в
+Generator/last_layer/fully_connected/MatMulMatMul Generator/third_layer/leaky_relu0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
ё
,Generator/last_layer/fully_connected/BiasAddBiasAdd+Generator/last_layer/fully_connected/MatMul.Generator/last_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
Ё
OGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:А*
dtype0
Ќ
EGenerator/last_layer/batch_normalization/gamma/Initializer/ones/ConstConst*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *  А?*
dtype0
Џ
?Generator/last_layer/batch_normalization/gamma/Initializer/onesFillOGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorEGenerator/last_layer/batch_normalization/gamma/Initializer/ones/Const*
_output_shapes	
:А*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
я
.Generator/last_layer/batch_normalization/gamma
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:А
¬
5Generator/last_layer/batch_normalization/gamma/AssignAssign.Generator/last_layer/batch_normalization/gamma?Generator/last_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
Ў
3Generator/last_layer/batch_normalization/gamma/readIdentity.Generator/last_layer/batch_normalization/gamma*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:А*
T0
№
OGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:А*
dtype0*
_output_shapes
:
ћ
EGenerator/last_layer/batch_normalization/beta/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ў
?Generator/last_layer/batch_normalization/beta/Initializer/zerosFillOGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorEGenerator/last_layer/batch_normalization/beta/Initializer/zeros/Const*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:А*
T0
Ё
-Generator/last_layer/batch_normalization/beta
VariableV2*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А
њ
4Generator/last_layer/batch_normalization/beta/AssignAssign-Generator/last_layer/batch_normalization/beta?Generator/last_layer/batch_normalization/beta/Initializer/zeros*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А*
use_locking(
’
2Generator/last_layer/batch_normalization/beta/readIdentity-Generator/last_layer/batch_normalization/beta*
_output_shapes	
:А*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
к
VGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB:А*
dtype0*
_output_shapes
:
Џ
LGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/ConstConst*
_output_shapes
: *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB
 *    *
dtype0
х
FGenerator/last_layer/batch_normalization/moving_mean/Initializer/zerosFillVGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/Const*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*

index_type0*
_output_shapes	
:А*
T0
л
4Generator/last_layer/batch_normalization/moving_mean
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
	container 
џ
;Generator/last_layer/batch_normalization/moving_mean/AssignAssign4Generator/last_layer/batch_normalization/moving_meanFGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:А
к
9Generator/last_layer/batch_normalization/moving_mean/readIdentity4Generator/last_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
_output_shapes	
:А
с
YGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB:А*
dtype0*
_output_shapes
:
б
OGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/ConstConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB
 *  А?*
dtype0*
_output_shapes
: 
В
IGenerator/last_layer/batch_normalization/moving_variance/Initializer/onesFillYGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorOGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/Const*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*

index_type0*
_output_shapes	
:А
у
8Generator/last_layer/batch_normalization/moving_variance
VariableV2*
shared_name *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
	container *
shape:А*
dtype0*
_output_shapes	
:А
к
?Generator/last_layer/batch_normalization/moving_variance/AssignAssign8Generator/last_layer/batch_normalization/moving_varianceIGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance
ц
=Generator/last_layer/batch_normalization/moving_variance/readIdentity8Generator/last_layer/batch_normalization/moving_variance*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
_output_shapes	
:А
}
8Generator/last_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
№
6Generator/last_layer/batch_normalization/batchnorm/addAdd=Generator/last_layer/batch_normalization/moving_variance/read8Generator/last_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:А
Я
8Generator/last_layer/batch_normalization/batchnorm/RsqrtRsqrt6Generator/last_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:А
“
6Generator/last_layer/batch_normalization/batchnorm/mulMul8Generator/last_layer/batch_normalization/batchnorm/Rsqrt3Generator/last_layer/batch_normalization/gamma/read*
_output_shapes	
:А*
T0
Ў
8Generator/last_layer/batch_normalization/batchnorm/mul_1Mul,Generator/last_layer/fully_connected/BiasAdd6Generator/last_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:€€€€€€€€€А*
T0
Ў
8Generator/last_layer/batch_normalization/batchnorm/mul_2Mul9Generator/last_layer/batch_normalization/moving_mean/read6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
—
6Generator/last_layer/batch_normalization/batchnorm/subSub2Generator/last_layer/batch_normalization/beta/read8Generator/last_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:А*
T0
д
8Generator/last_layer/batch_normalization/batchnorm/add_1Add8Generator/last_layer/batch_normalization/batchnorm/mul_16Generator/last_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:€€€€€€€€€А*
T0
j
%Generator/last_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
Њ
#Generator/last_layer/leaky_relu/mulMul%Generator/last_layer/leaky_relu/alpha8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:€€€€€€€€€А*
T0
Љ
Generator/last_layer/leaky_reluMaximum#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:€€€€€€€€€А*
T0
љ
<Generator/fake_image/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
ѓ
:Generator/fake_image/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zхkљ*
dtype0
ѓ
:Generator/fake_image/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zхk=*
dtype0*
_output_shapes
: 
Ф
DGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniformRandomUniform<Generator/fake_image/kernel/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
seed2 *
dtype0* 
_output_shapes
:
АР*

seed 
К
:Generator/fake_image/kernel/Initializer/random_uniform/subSub:Generator/fake_image/kernel/Initializer/random_uniform/max:Generator/fake_image/kernel/Initializer/random_uniform/min*.
_class$
" loc:@Generator/fake_image/kernel*
_output_shapes
: *
T0
Ю
:Generator/fake_image/kernel/Initializer/random_uniform/mulMulDGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniform:Generator/fake_image/kernel/Initializer/random_uniform/sub*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
АР*
T0
Р
6Generator/fake_image/kernel/Initializer/random_uniformAdd:Generator/fake_image/kernel/Initializer/random_uniform/mul:Generator/fake_image/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
АР
√
Generator/fake_image/kernel
VariableV2*
	container *
shape:
АР*
dtype0* 
_output_shapes
:
АР*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel
Е
"Generator/fake_image/kernel/AssignAssignGenerator/fake_image/kernel6Generator/fake_image/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
АР*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel
§
 Generator/fake_image/kernel/readIdentityGenerator/fake_image/kernel* 
_output_shapes
:
АР*
T0*.
_class$
" loc:@Generator/fake_image/kernel
®
+Generator/fake_image/bias/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueBР*    *
dtype0*
_output_shapes	
:Р
µ
Generator/fake_image/bias
VariableV2*
dtype0*
_output_shapes	
:Р*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:Р
п
 Generator/fake_image/bias/AssignAssignGenerator/fake_image/bias+Generator/fake_image/bias/Initializer/zeros*
_output_shapes	
:Р*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
Щ
Generator/fake_image/bias/readIdentityGenerator/fake_image/bias*
_output_shapes	
:Р*
T0*,
_class"
 loc:@Generator/fake_image/bias
Ѕ
Generator/fake_image/MatMulMatMulGenerator/last_layer/leaky_relu Generator/fake_image/kernel/read*(
_output_shapes
:€€€€€€€€€Р*
transpose_a( *
transpose_b( *
T0
Ѓ
Generator/fake_image/BiasAddBiasAddGenerator/fake_image/MatMulGenerator/fake_image/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€Р*
T0
r
Generator/fake_image/TanhTanhGenerator/fake_image/BiasAdd*(
_output_shapes
:€€€€€€€€€Р*
T0
z
Discriminator/real_inPlaceholder*
dtype0*(
_output_shapes
:€€€€€€€€€Р*
shape:€€€€€€€€€Р
з
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ў
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HYЛљ*
dtype0*
_output_shapes
: 
ў
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HYЛ=
”
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
РА*

seed *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
ё
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
т
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
РА
д
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
РА
н
0Discriminator/first_layer/fully_connected/kernel
VariableV2*
shape:
РА*
dtype0* 
_output_shapes
:
РА*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container 
ў
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
РА
г
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
РА*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
“
@Discriminator/first_layer/fully_connected/bias/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
я
.Discriminator/first_layer/fully_connected/bias
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
√
5Discriminator/first_layer/fully_connected/bias/AssignAssign.Discriminator/first_layer/fully_connected/bias@Discriminator/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
Ў
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:А*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
б
0Discriminator/first_layer/fully_connected/MatMulMatMulDiscriminator/real_in5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
н
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
o
*Discriminator/first_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
Ѕ
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
њ
$Discriminator/first_layer/leaky_reluMaximum(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
й
RDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
џ
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *уµљ*
dtype0*
_output_shapes
: 
џ
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *уµ=*
dtype0*
_output_shapes
: 
÷
ZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformRDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
в
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
_output_shapes
: 
ц
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
и
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
п
1Discriminator/second_layer/fully_connected/kernel
VariableV2* 
_output_shapes
:
АА*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
АА*
dtype0
Ё
8Discriminator/second_layer/fully_connected/kernel/AssignAssign1Discriminator/second_layer/fully_connected/kernelLDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
ж
6Discriminator/second_layer/fully_connected/kernel/readIdentity1Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
‘
ADiscriminator/second_layer/fully_connected/bias/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
б
/Discriminator/second_layer/fully_connected/bias
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container 
«
6Discriminator/second_layer/fully_connected/bias/AssignAssign/Discriminator/second_layer/fully_connected/biasADiscriminator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
џ
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:А
т
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
р
2Discriminator/second_layer/fully_connected/BiasAddBiasAdd1Discriminator/second_layer/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
p
+Discriminator/second_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
ƒ
)Discriminator/second_layer/leaky_relu/mulMul+Discriminator/second_layer/leaky_relu/alpha2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
¬
%Discriminator/second_layer/leaky_reluMaximum)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
є
:Discriminator/prob/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ђ
8Discriminator/prob/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *IvЊ*
dtype0*
_output_shapes
: 
Ђ
8Discriminator/prob/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
Н
BDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Discriminator/prob/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
seed2 
В
8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
: 
Х
8Discriminator/prob/kernel/Initializer/random_uniform/mulMulBDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniform8Discriminator/prob/kernel/Initializer/random_uniform/sub*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	А*
T0
З
4Discriminator/prob/kernel/Initializer/random_uniformAdd8Discriminator/prob/kernel/Initializer/random_uniform/mul8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	А
љ
Discriminator/prob/kernel
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel
ь
 Discriminator/prob/kernel/AssignAssignDiscriminator/prob/kernel4Discriminator/prob/kernel/Initializer/random_uniform*
_output_shapes
:	А*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(
Э
Discriminator/prob/kernel/readIdentityDiscriminator/prob/kernel*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	А
Ґ
)Discriminator/prob/bias/Initializer/zerosConst*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0
ѓ
Discriminator/prob/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container *
shape:
ж
Discriminator/prob/bias/AssignAssignDiscriminator/prob/bias)Discriminator/prob/bias/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
Т
Discriminator/prob/bias/readIdentityDiscriminator/prob/bias**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:*
T0
¬
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
І
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
з
2Discriminator/first_layer_1/fully_connected/MatMulMatMulGenerator/fake_image/Tanh5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
с
3Discriminator/first_layer_1/fully_connected/BiasAddBiasAdd2Discriminator/first_layer_1/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
q
,Discriminator/first_layer_1/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
«
*Discriminator/first_layer_1/leaky_relu/mulMul,Discriminator/first_layer_1/leaky_relu/alpha3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
≈
&Discriminator/first_layer_1/leaky_reluMaximum*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
ц
3Discriminator/second_layer_1/fully_connected/MatMulMatMul&Discriminator/first_layer_1/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
ф
4Discriminator/second_layer_1/fully_connected/BiasAddBiasAdd3Discriminator/second_layer_1/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
r
-Discriminator/second_layer_1/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
 
+Discriminator/second_layer_1/leaky_relu/mulMul-Discriminator/second_layer_1/leaky_relu/alpha4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
»
'Discriminator/second_layer_1/leaky_reluMaximum+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
∆
Discriminator/prob_1/MatMulMatMul'Discriminator/second_layer_1/leaky_reluDiscriminator/prob/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
Ђ
Discriminator/prob_1/BiasAddBiasAddDiscriminator/prob_1/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
i
ones_like/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
T
ones_like/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
w
	ones_likeFillones_like/Shapeones_like/Const*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
s
logistic_loss/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Т
logistic_loss/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
Ґ
logistic_loss/SelectSelectlogistic_loss/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:€€€€€€€€€*
T0
f
logistic_loss/NegNegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Э
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/NegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
q
logistic_loss/mulMulDiscriminator/prob/BiasAdd	ones_like*
T0*'
_output_shapes
:€€€€€€€€€
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0*'
_output_shapes
:€€€€€€€€€
b
logistic_loss/ExpExplogistic_loss/Select_1*
T0*'
_output_shapes
:€€€€€€€€€
a
logistic_loss/Log1pLog1plogistic_loss/Exp*'
_output_shapes
:€€€€€€€€€*
T0
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*'
_output_shapes
:€€€€€€€€€*
T0
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
`
MeanMeanlogistic_lossConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
g

zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
w
logistic_loss_1/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Ш
logistic_loss_1/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
™
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
j
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
•
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
v
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAdd
zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
y
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*'
_output_shapes
:€€€€€€€€€*
T0
f
logistic_loss_1/ExpExplogistic_loss_1/Select_1*'
_output_shapes
:€€€€€€€€€*
T0
e
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*
T0*'
_output_shapes
:€€€€€€€€€
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*
T0*'
_output_shapes
:€€€€€€€€€
X
Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
f
Mean_1Meanlogistic_loss_1Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
9
addAddMeanMean_1*
_output_shapes
: *
T0
i
discriminator_loss/tagConst*
dtype0*
_output_shapes
: *#
valueB Bdiscriminator_loss
d
discriminator_lossHistogramSummarydiscriminator_loss/tagadd*
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
 *  А?*
dtype0*
_output_shapes
: 
}
ones_like_1Fillones_like_1/Shapeones_like_1/Const*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
w
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
Ш
logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:€€€€€€€€€*
T0
™
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
j
logistic_loss_2/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
•
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
w
logistic_loss_2/mulMulDiscriminator/prob_1/BiasAddones_like_1*
T0*'
_output_shapes
:€€€€€€€€€
y
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*
T0*'
_output_shapes
:€€€€€€€€€
f
logistic_loss_2/ExpExplogistic_loss_2/Select_1*
T0*'
_output_shapes
:€€€€€€€€€
e
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*
T0*'
_output_shapes
:€€€€€€€€€
t
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*'
_output_shapes
:€€€€€€€€€*
T0
X
Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
f
Mean_2Meanlogistic_loss_2Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
a
generator_loss/tagConst*
_output_shapes
: *
valueB Bgenerator_loss*
dtype0
_
generator_lossHistogramSummarygenerator_loss/tagMean_2*
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
 *  А?*
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
±
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Fill$^gradients/add_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
_output_shapes
: *
T0
≥
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
≠
gradients/Mean_grad/ReshapeReshape+gradients/add_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
f
gradients/Mean_grad/ShapeShapelogistic_loss*
T0*
out_type0*
_output_shapes
:
Ь
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
h
gradients/Mean_grad/Shape_1Shapelogistic_loss*
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
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ц
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
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
А
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
М
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
t
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
≥
gradients/Mean_1_grad/ReshapeReshape-gradients/add_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
j
gradients/Mean_1_grad/ShapeShapelogistic_loss_1*
_output_shapes
:*
T0*
out_type0
Ґ
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€*

Tmultiples0
l
gradients/Mean_1_grad/Shape_1Shapelogistic_loss_1*
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
gradients/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ь
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
†
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
И
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
Ж
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
В
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
Т
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*'
_output_shapes
:€€€€€€€€€*
T0
s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0*
_output_shapes
:
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
out_type0*
_output_shapes
:*
T0
“
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Є
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
µ
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
Љ
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ї
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Е
-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
В
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*'
_output_shapes
:€€€€€€€€€
И
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1
w
$gradients/logistic_loss_1_grad/ShapeShapelogistic_loss_1/sub*
T0*
out_type0*
_output_shapes
:
{
&gradients/logistic_loss_1_grad/Shape_1Shapelogistic_loss_1/Log1p*
_output_shapes
:*
T0*
out_type0
Ў
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Њ
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ї
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
¬
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Л
/gradients/logistic_loss_1_grad/tuple/group_depsNoOp'^gradients/logistic_loss_1_grad/Reshape)^gradients/logistic_loss_1_grad/Reshape_1
К
7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Р
9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
z
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0*
_output_shapes
:
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
T0*
out_type0*
_output_shapes
:
ё
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Џ
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ѕ
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
ё
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
≈
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
С
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
Т
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Ш
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
І
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ь
&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:€€€€€€€€€
Х
-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:€€€€€€€€€
«
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€
~
(gradients/logistic_loss_1/sub_grad/ShapeShapelogistic_loss_1/Select*
T0*
out_type0*
_output_shapes
:
}
*gradients/logistic_loss_1/sub_grad/Shape_1Shapelogistic_loss_1/mul*
_output_shapes
:*
T0*
out_type0
д
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
«
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
д
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
Ћ
,gradients/logistic_loss_1/sub_grad/Reshape_1Reshape&gradients/logistic_loss_1/sub_grad/Neg*gradients/logistic_loss_1/sub_grad/Shape_1*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
Ч
3gradients/logistic_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/sub_grad/Reshape-^gradients/logistic_loss_1/sub_grad/Reshape_1
Ъ
;gradients/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/sub_grad/Reshape4^gradients/logistic_loss_1/sub_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss_1/sub_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
†
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1
Ђ
*gradients/logistic_loss_1/Log1p_grad/add/xConst:^gradients/logistic_loss_1_grad/tuple/control_dependency_1*
_output_shapes
: *
valueB
 *  А?*
dtype0
Ґ
(gradients/logistic_loss_1/Log1p_grad/addAdd*gradients/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*'
_output_shapes
:€€€€€€€€€*
T0
Щ
/gradients/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*
T0*'
_output_shapes
:€€€€€€€€€
Ќ
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€*
T0
Й
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
н
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
п
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€
Ш
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
Ь
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
Ґ
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:€€€€€€€€€
А
&gradients/logistic_loss/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
q
(gradients/logistic_loss/mul_grad/Shape_1Shape	ones_like*
out_type0*
_output_shapes
:*
T0
ё
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
•
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*'
_output_shapes
:€€€€€€€€€*
T0
…
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ѕ
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Є
&gradients/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€*
T0
ѕ
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
«
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
С
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
Т
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Ш
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1
Ш
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*'
_output_shapes
:€€€€€€€€€*
T0
Н
0gradients/logistic_loss_1/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
х
,gradients/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual;gradients/logistic_loss_1/sub_grad/tuple/control_dependency0gradients/logistic_loss_1/Select_grad/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
ч
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€*
T0
Ю
6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
§
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
™
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_grad/Select_1*'
_output_shapes
:€€€€€€€€€
Д
(gradients/logistic_loss_1/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
t
*gradients/logistic_loss_1/mul_grad/Shape_1Shape
zeros_like*
_output_shapes
:*
T0*
out_type0
д
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
™
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1
zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
ѕ
&gradients/logistic_loss_1/mul_grad/SumSum&gradients/logistic_loss_1/mul_grad/Mul8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
«
*gradients/logistic_loss_1/mul_grad/ReshapeReshape&gradients/logistic_loss_1/mul_grad/Sum(gradients/logistic_loss_1/mul_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
Њ
(gradients/logistic_loss_1/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€
’
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ќ
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Ч
3gradients/logistic_loss_1/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/mul_grad/Reshape-^gradients/logistic_loss_1/mul_grad/Reshape_1
Ъ
;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/mul_grad/Reshape4^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€
†
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€*
T0
Ю
&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0*'
_output_shapes
:€€€€€€€€€
В
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*'
_output_shapes
:€€€€€€€€€*
T0
№
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
ё
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:€€€€€€€€€
Ю
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
§
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select*'
_output_shapes
:€€€€€€€€€
™
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1*'
_output_shapes
:€€€€€€€€€
Ж
2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*
T0*'
_output_shapes
:€€€€€€€€€
д
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*'
_output_shapes
:€€€€€€€€€*
T0
ж
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*'
_output_shapes
:€€€€€€€€€*
T0
§
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
ђ
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select*'
_output_shapes
:€€€€€€€€€
≤
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1
Э
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€*
T0
°
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€
с
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:€€€€€€€€€
Р
5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
_output_shapes
:*
T0
Л
:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN6^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad
М
Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
≥
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
э
gradients/AddN_1AddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N*'
_output_shapes
:€€€€€€€€€*
T0
Ф
7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
_output_shapes
:*
T0*
data_formatNHWC
С
<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_18^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
Ф
Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
ї
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
ц
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
ц
1gradients/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А*
transpose_a(*
transpose_b( *
T0
І
9gradients/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp0^gradients/Discriminator/prob/MatMul_grad/MatMul2^gradients/Discriminator/prob/MatMul_grad/MatMul_1
±
Agradients/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity/gradients/Discriminator/prob/MatMul_grad/MatMul:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/Discriminator/prob/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Ѓ
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	А
ъ
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
ь
3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	А*
transpose_a(*
transpose_b( 
≠
;gradients/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp2^gradients/Discriminator/prob_1/MatMul_grad/MatMul4^gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
є
Cgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/Discriminator/prob_1/MatMul_grad/MatMul<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
ґ
Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	А
Ю
gradients/AddN_2AddNDgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
£
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ѓ
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
љ
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Е
@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
З
:gradients/Discriminator/second_layer/leaky_relu_grad/zerosFill<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
г
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
Ъ
Jgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Discriminator/second_layer/leaky_relu_grad/Shape<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Ї
;gradients/Discriminator/second_layer/leaky_relu_grad/SelectSelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency:gradients/Discriminator/second_layer/leaky_relu_grad/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
Љ
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
И
8gradients/Discriminator/second_layer/leaky_relu_grad/SumSum;gradients/Discriminator/second_layer/leaky_relu_grad/SelectJgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ю
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
О
:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1Lgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Д
>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ќ
Egradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp=^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape?^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
г
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
й
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
І
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
≤
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ѕ
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
З
Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosFill>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
й
Cgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
†
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
¬
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
ƒ
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
О
:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Д
>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape:gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ф
<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
К
@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
”
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
л
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
с
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Э
gradients/AddN_3AddNCgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1*
N*
_output_shapes
:	А*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1
Б
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
≤
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
¶
Ngradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
щ
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulMulMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
С
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ш
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ф
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
Ч
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Pgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Р
Bgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
ў
Igradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpA^gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeC^gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
б
Qgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeJ^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
щ
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
Г
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
ґ
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
ђ
Pgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
€
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Ч
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ю
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ъ
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
Э
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Rgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ц
Dgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1Reshape@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
я
Kgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeE^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
й
Sgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeL^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
Б
Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1L^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ќ
gradients/AddN_4AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
Ђ
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:А
љ
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
”
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ф
\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
”
gradients/AddN_5AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
≠
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
data_formatNHWC*
_output_shapes	
:А*
T0
Ѕ
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ў
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ь
^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
Њ
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
¶
Igradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_reluZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
п
Qgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpH^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulJ^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
С
Ygradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityGgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulR^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
П
[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
¬
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
ђ
Kgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
х
Sgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulL^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
Щ
[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulT^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul
Ч
]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
з
gradients/AddN_6AddN\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:А
°
9gradients/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
ђ
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
‘
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2ShapeYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Д
?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Д
9gradients/Discriminator/first_layer/leaky_relu_grad/zerosFill;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:€€€€€€€€€А*
T0*

index_type0
а
@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Ч
Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ѕ
:gradients/Discriminator/first_layer/leaky_relu_grad/SelectSelect@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency9gradients/Discriminator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
—
<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Select@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqual9gradients/Discriminator/first_layer/leaky_relu_grad/zerosYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Е
7gradients/Discriminator/first_layer/leaky_relu_grad/SumSum:gradients/Discriminator/first_layer/leaky_relu_grad/SelectIgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ы
;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape7gradients/Discriminator/first_layer/leaky_relu_grad/Sum9gradients/Discriminator/first_layer/leaky_relu_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
Л
9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Kgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Б
=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dgradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp<^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape>^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
я
Lgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeE^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
•
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
∞
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ў
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Ж
Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
К
;gradients/Discriminator/first_layer_1/leaky_relu_grad/zerosFill=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
ж
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Э
Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
„
<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
ў
>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Л
9gradients/Discriminator/first_layer_1/leaky_relu_grad/SumSum<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectKgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Б
=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
С
;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
З
?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
–
Fgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp>^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape@^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
з
Ngradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeG^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*P
_classF
DBloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
н
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
ж
gradients/AddN_7AddN[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
АА
А
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
∞
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
£
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ц
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
О
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
х
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
с
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
Ф
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Ogradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
Agradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
÷
Hgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp@^gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeB^gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
Ё
Pgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeI^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape
х
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
В
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
і
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
©
Ogradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ь
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
Ф
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulOgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ы
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
ч
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Qgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
У
Cgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1Reshape?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
№
Jgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeD^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
е
Rgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeK^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
э
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
 
gradients/AddN_8AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
™
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:А
ї
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
–
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Р
[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradR^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
–
gradients/AddN_9AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:€€€€€€€€€А*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
ђ
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
data_formatNHWC*
_output_shapes	
:А*
T0
њ
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
÷
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ш
]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ї
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€Р*
transpose_a( *
transpose_b(
Х
Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/real_inYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
РА*
transpose_a(*
transpose_b( 
м
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
Н
Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€Р
Л
Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
РА
њ
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€Р*
transpose_a( *
transpose_b(
Э
Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
РА*
transpose_a(*
transpose_b( *
T0
т
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
Х
Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€Р*
T0
У
\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1S^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
РА*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
е
gradients/AddN_10AddN[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes	
:А*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
д
gradients/AddN_11AddNZgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
РА*
T0
°
beta1_power/initial_valueConst*
_output_shapes
: *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *fff?*
dtype0
≤
beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
—
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
Н
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
°
beta2_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
≤
beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container 
—
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
Н
beta2_power/readIdentitybeta2_power*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
н
WDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     
„
MDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
щ
GDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillWDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorMDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
РА*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0
т
5Discriminator/first_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
РА*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
РА
я
<Discriminator/first_layer/fully_connected/kernel/Adam/AssignAssign5Discriminator/first_layer/fully_connected/kernel/AdamGDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
РА
н
:Discriminator/first_layer/fully_connected/kernel/Adam/readIdentity5Discriminator/first_layer/fully_connected/kernel/Adam* 
_output_shapes
:
РА*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
п
YDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ў
ODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
€
IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillYDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
РА*
T0
ф
7Discriminator/first_layer/fully_connected/kernel/Adam_1
VariableV2*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
РА*
dtype0* 
_output_shapes
:
РА*
shared_name 
е
>Discriminator/first_layer/fully_connected/kernel/Adam_1/AssignAssign7Discriminator/first_layer/fully_connected/kernel/Adam_1IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
РА*
use_locking(
с
<Discriminator/first_layer/fully_connected/kernel/Adam_1/readIdentity7Discriminator/first_layer/fully_connected/kernel/Adam_1*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
РА
„
EDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
д
3Discriminator/first_layer/fully_connected/bias/Adam
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
“
:Discriminator/first_layer/fully_connected/bias/Adam/AssignAssign3Discriminator/first_layer/fully_connected/bias/AdamEDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
в
8Discriminator/first_layer/fully_connected/bias/Adam/readIdentity3Discriminator/first_layer/fully_connected/bias/Adam*
_output_shapes	
:А*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
ў
GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ж
5Discriminator/first_layer/fully_connected/bias/Adam_1
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
Ў
<Discriminator/first_layer/fully_connected/bias/Adam_1/AssignAssign5Discriminator/first_layer/fully_connected/bias/Adam_1GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(
ж
:Discriminator/first_layer/fully_connected/bias/Adam_1/readIdentity5Discriminator/first_layer/fully_connected/bias/Adam_1*
_output_shapes	
:А*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
п
XDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ў
NDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0
э
HDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillXDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorNDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА
ф
6Discriminator/second_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
АА
г
=Discriminator/second_layer/fully_connected/kernel/Adam/AssignAssign6Discriminator/second_layer/fully_connected/kernel/AdamHDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
р
;Discriminator/second_layer/fully_connected/kernel/Adam/readIdentity6Discriminator/second_layer/fully_connected/kernel/Adam* 
_output_shapes
:
АА*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
с
ZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0
џ
PDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorPDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0
ц
8Discriminator/second_layer/fully_connected/kernel/Adam_1
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
й
?Discriminator/second_layer/fully_connected/kernel/Adam_1/AssignAssign8Discriminator/second_layer/fully_connected/kernel/Adam_1JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0
ф
=Discriminator/second_layer/fully_connected/kernel/Adam_1/readIdentity8Discriminator/second_layer/fully_connected/kernel/Adam_1*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
ў
FDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ж
4Discriminator/second_layer/fully_connected/bias/Adam
VariableV2*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
÷
;Discriminator/second_layer/fully_connected/bias/Adam/AssignAssign4Discriminator/second_layer/fully_connected/bias/AdamFDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(
е
9Discriminator/second_layer/fully_connected/bias/Adam/readIdentity4Discriminator/second_layer/fully_connected/bias/Adam*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:А
џ
HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
и
6Discriminator/second_layer/fully_connected/bias/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container 
№
=Discriminator/second_layer/fully_connected/bias/Adam_1/AssignAssign6Discriminator/second_layer/fully_connected/bias/Adam_1HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
й
;Discriminator/second_layer/fully_connected/bias/Adam_1/readIdentity6Discriminator/second_layer/fully_connected/bias/Adam_1*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:А
µ
0Discriminator/prob/kernel/Adam/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
¬
Discriminator/prob/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	А*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	А
В
%Discriminator/prob/kernel/Adam/AssignAssignDiscriminator/prob/kernel/Adam0Discriminator/prob/kernel/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	А
І
#Discriminator/prob/kernel/Adam/readIdentityDiscriminator/prob/kernel/Adam*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	А
Ј
2Discriminator/prob/kernel/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
ƒ
 Discriminator/prob/kernel/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
И
'Discriminator/prob/kernel/Adam_1/AssignAssign Discriminator/prob/kernel/Adam_12Discriminator/prob/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	А*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(
Ђ
%Discriminator/prob/kernel/Adam_1/readIdentity Discriminator/prob/kernel/Adam_1*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	А
І
.Discriminator/prob/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    
і
Discriminator/prob/bias/Adam
VariableV2*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0*
_output_shapes
:
х
#Discriminator/prob/bias/Adam/AssignAssignDiscriminator/prob/bias/Adam.Discriminator/prob/bias/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
Ь
!Discriminator/prob/bias/Adam/readIdentityDiscriminator/prob/bias/Adam*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
©
0Discriminator/prob/bias/Adam_1/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
ґ
Discriminator/prob/bias/Adam_1
VariableV2*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0
ы
%Discriminator/prob/bias/Adam_1/AssignAssignDiscriminator/prob/bias/Adam_10Discriminator/prob/bias/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
†
#Discriminator/prob/bias/Adam_1/readIdentityDiscriminator/prob/bias/Adam_1*
_output_shapes
:*
T0**
_class 
loc:@Discriminator/prob/bias
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *ЈQ9
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
э
FAdam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam0Discriminator/first_layer/fully_connected/kernel5Discriminator/first_layer/fully_connected/kernel/Adam7Discriminator/first_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_11* 
_output_shapes
:
РА*
use_locking( *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
use_nesterov( 
о
DAdam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam.Discriminator/first_layer/fully_connected/bias3Discriminator/first_layer/fully_connected/bias/Adam5Discriminator/first_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_10*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:А
Б
GAdam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernel6Discriminator/second_layer/fully_connected/kernel/Adam8Discriminator/second_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
АА
т
EAdam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/bias4Discriminator/second_layer/fully_connected/bias/Adam6Discriminator/second_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
И
/Adam/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernelDiscriminator/prob/kernel/Adam Discriminator/prob/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
use_nesterov( *
_output_shapes
:	А*
use_locking( 
щ
-Adam/update_Discriminator/prob/bias/ApplyAdam	ApplyAdamDiscriminator/prob/biasDiscriminator/prob/bias/AdamDiscriminator/prob/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2**
_class 
loc:@Discriminator/prob/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
Х
Adam/mulMulbeta1_power/read
Adam/beta1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
є
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
Ч

Adam/mul_1Mulbeta2_power/read
Adam/beta2E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
љ
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
Ѓ
AdamNoOp^Adam/Assign^Adam/Assign_1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
v
%gradients_1/Mean_2_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ъ
gradients_1/Mean_2_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
l
gradients_1/Mean_2_grad/ShapeShapelogistic_loss_2*
T0*
out_type0*
_output_shapes
:
®
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
n
gradients_1/Mean_2_grad/Shape_1Shapelogistic_loss_2*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_2_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ґ
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_2_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
¶
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
c
!gradients_1/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
О
gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
Ж
gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
Ш
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
y
&gradients_1/logistic_loss_2_grad/ShapeShapelogistic_loss_2/sub*
_output_shapes
:*
T0*
out_type0
}
(gradients_1/logistic_loss_2_grad/Shape_1Shapelogistic_loss_2/Log1p*
T0*
out_type0*
_output_shapes
:
ё
6gradients_1/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_2_grad/Shape(gradients_1/logistic_loss_2_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ƒ
$gradients_1/logistic_loss_2_grad/SumSumgradients_1/Mean_2_grad/truediv6gradients_1/logistic_loss_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ѕ
(gradients_1/logistic_loss_2_grad/ReshapeReshape$gradients_1/logistic_loss_2_grad/Sum&gradients_1/logistic_loss_2_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
»
&gradients_1/logistic_loss_2_grad/Sum_1Sumgradients_1/Mean_2_grad/truediv8gradients_1/logistic_loss_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
«
*gradients_1/logistic_loss_2_grad/Reshape_1Reshape&gradients_1/logistic_loss_2_grad/Sum_1(gradients_1/logistic_loss_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
С
1gradients_1/logistic_loss_2_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_2_grad/Reshape+^gradients_1/logistic_loss_2_grad/Reshape_1
Т
9gradients_1/logistic_loss_2_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_2_grad/Reshape2^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_2_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Ш
;gradients_1/logistic_loss_2_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_2_grad/Reshape_12^gradients_1/logistic_loss_2_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*=
_class3
1/loc:@gradients_1/logistic_loss_2_grad/Reshape_1
А
*gradients_1/logistic_loss_2/sub_grad/ShapeShapelogistic_loss_2/Select*
_output_shapes
:*
T0*
out_type0

,gradients_1/logistic_loss_2/sub_grad/Shape_1Shapelogistic_loss_2/mul*
_output_shapes
:*
T0*
out_type0
к
:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/sub_grad/Shape,gradients_1/logistic_loss_2/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ж
(gradients_1/logistic_loss_2/sub_grad/SumSum9gradients_1/logistic_loss_2_grad/tuple/control_dependency:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ќ
,gradients_1/logistic_loss_2/sub_grad/ReshapeReshape(gradients_1/logistic_loss_2/sub_grad/Sum*gradients_1/logistic_loss_2/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
к
*gradients_1/logistic_loss_2/sub_grad/Sum_1Sum9gradients_1/logistic_loss_2_grad/tuple/control_dependency<gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
~
(gradients_1/logistic_loss_2/sub_grad/NegNeg*gradients_1/logistic_loss_2/sub_grad/Sum_1*
_output_shapes
:*
T0
—
.gradients_1/logistic_loss_2/sub_grad/Reshape_1Reshape(gradients_1/logistic_loss_2/sub_grad/Neg,gradients_1/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Э
5gradients_1/logistic_loss_2/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/sub_grad/Reshape/^gradients_1/logistic_loss_2/sub_grad/Reshape_1
Ґ
=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/sub_grad/Reshape6^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/sub_grad/Reshape*'
_output_shapes
:€€€€€€€€€
®
?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/sub_grad/Reshape_16^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/sub_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
ѓ
,gradients_1/logistic_loss_2/Log1p_grad/add/xConst<^gradients_1/logistic_loss_2_grad/tuple/control_dependency_1*
valueB
 *  А?*
dtype0*
_output_shapes
: 
¶
*gradients_1/logistic_loss_2/Log1p_grad/addAdd,gradients_1/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*
T0*'
_output_shapes
:€€€€€€€€€
Э
1gradients_1/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_2/Log1p_grad/add*
T0*'
_output_shapes
:€€€€€€€€€
”
*gradients_1/logistic_loss_2/Log1p_grad/mulMul;gradients_1/logistic_loss_2_grad/tuple/control_dependency_11gradients_1/logistic_loss_2/Log1p_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€*
T0
П
2gradients_1/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
ы
.gradients_1/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_2/Select_grad/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
э
0gradients_1/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients_1/logistic_loss_2/Select_grad/zeros_like=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€
§
8gradients_1/logistic_loss_2/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_2/Select_grad/Select1^gradients_1/logistic_loss_2/Select_grad/Select_1
ђ
@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_2/Select_grad/Select9^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
≤
Bgradients_1/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_2/Select_grad/Select_19^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_grad/Select_1*'
_output_shapes
:€€€€€€€€€
Ж
*gradients_1/logistic_loss_2/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
w
,gradients_1/logistic_loss_2/mul_grad/Shape_1Shapeones_like_1*
T0*
out_type0*
_output_shapes
:
к
:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/mul_grad/Shape,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ѓ
(gradients_1/logistic_loss_2/mul_grad/MulMul?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1ones_like_1*'
_output_shapes
:€€€€€€€€€*
T0
’
(gradients_1/logistic_loss_2/mul_grad/SumSum(gradients_1/logistic_loss_2/mul_grad/Mul:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ќ
,gradients_1/logistic_loss_2/mul_grad/ReshapeReshape(gradients_1/logistic_loss_2/mul_grad/Sum*gradients_1/logistic_loss_2/mul_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
¬
*gradients_1/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€
џ
*gradients_1/logistic_loss_2/mul_grad/Sum_1Sum*gradients_1/logistic_loss_2/mul_grad/Mul_1<gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
”
.gradients_1/logistic_loss_2/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_2/mul_grad/Sum_1,gradients_1/logistic_loss_2/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Э
5gradients_1/logistic_loss_2/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/mul_grad/Reshape/^gradients_1/logistic_loss_2/mul_grad/Reshape_1
Ґ
=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/mul_grad/Reshape6^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€
®
?gradients_1/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/mul_grad/Reshape_16^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€*
T0
Ґ
(gradients_1/logistic_loss_2/Exp_grad/mulMul*gradients_1/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*
T0*'
_output_shapes
:€€€€€€€€€
И
4gradients_1/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*'
_output_shapes
:€€€€€€€€€*
T0
к
0gradients_1/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual(gradients_1/logistic_loss_2/Exp_grad/mul4gradients_1/logistic_loss_2/Select_1_grad/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
м
2gradients_1/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual4gradients_1/logistic_loss_2/Select_1_grad/zeros_like(gradients_1/logistic_loss_2/Exp_grad/mul*
T0*'
_output_shapes
:€€€€€€€€€
™
:gradients_1/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_2/Select_1_grad/Select3^gradients_1/logistic_loss_2/Select_1_grad/Select_1
і
Bgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_2/Select_1_grad/Select;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_1_grad/Select*'
_output_shapes
:€€€€€€€€€
Ї
Dgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_2/Select_1_grad/Select_1;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/logistic_loss_2/Select_1_grad/Select_1*'
_output_shapes
:€€€€€€€€€
•
(gradients_1/logistic_loss_2/Neg_grad/NegNegBgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€
З
gradients_1/AddNAddN@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_2/Neg_grad/Neg*
N*'
_output_shapes
:€€€€€€€€€*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select
Ц
9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
data_formatNHWC*
_output_shapes
:*
T0
Х
>gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN:^gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
Ъ
Fgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
√
Hgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ю
3gradients_1/Discriminator/prob_1/MatMul_grad/MatMulMatMulFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
А
5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	А*
transpose_a(*
transpose_b( 
≥
=gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul6^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1
Ѕ
Egradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Discriminator/prob_1/MatMul_grad/MatMul>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Њ
Ggradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	А
©
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
і
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
≈
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Й
Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
У
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosFill@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
л
Egradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
¶
Ngradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
 
?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
ћ
Agradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
Ф
<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SumSum?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectNgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
К
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ъ
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1SumAgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1Pgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Р
Bgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ў
Igradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOpA^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeC^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
у
Qgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeJ^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
щ
Sgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityBgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1J^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Е
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Є
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
≤
Rgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Г
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Э
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulRgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Д
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ю
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
£
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Tgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ь
Fgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
е
Mgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpE^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeG^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
с
Ugradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeN^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
Й
Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityFgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1N^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
џ
gradients_1/AddN_1AddNSgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N
±
Qgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:А
«
Vgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1R^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
б
^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1W^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
§
`gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityQgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
∆
Kgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
∞
Mgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
ы
Ugradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpL^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulN^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
°
]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityKgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulV^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*^
_classT
RPloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А*
T0
Я
_gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityMgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1V^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
І
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
≤
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
№
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
И
Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zerosFill?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:€€€€€€€€€А*
T0
и
Dgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
£
Mgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
я
>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
б
@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
С
;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SumSum>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectMgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
З
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
Ч
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1Ogradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
Agradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
÷
Hgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp@^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeB^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
п
Pgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeI^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
х
Rgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityAgradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1I^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
Д
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
ґ
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
ѓ
Qgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
А
?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulQgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Б
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ы
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
†
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Sgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Щ
Egradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
в
Lgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpD^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeF^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
н
Tgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeM^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*V
_classL
JHloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
Е
Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityEgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1M^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ў
gradients_1/AddN_2AddNRgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
∞
Pgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:А
≈
Ugradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2Q^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ё
]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2V^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
†
_gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradV^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
√
Jgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€Р*
transpose_a( *
transpose_b(
°
Lgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
РА*
transpose_a(*
transpose_b( *
T0
ш
Tgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpK^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulM^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
Э
\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityJgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulU^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€Р
Ы
^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityLgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1U^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
РА
л
3gradients_1/Generator/fake_image/Tanh_grad/TanhGradTanhGradGenerator/fake_image/Tanh\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€Р
Ї
9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:Р
Є
>gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad4^gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
√
Fgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€Р*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
ƒ
Hgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:Р*
T0*L
_classB
@>loc:@gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad
А
3gradients_1/Generator/fake_image/MatMul_grad/MatMulMatMulFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency Generator/fake_image/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
щ
5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1MatMulGenerator/last_layer/leaky_reluFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АР*
transpose_a(*
transpose_b( *
T0
≥
=gradients_1/Generator/fake_image/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Generator/fake_image/MatMul_grad/MatMul6^gradients_1/Generator/fake_image/MatMul_grad/MatMul_1
Ѕ
Egradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/MatMul_grad/MatMul>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul
њ
Ggradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul_1* 
_output_shapes
:
АР
Щ
6gradients_1/Generator/last_layer/leaky_relu_grad/ShapeShape#Generator/last_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
∞
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
љ
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2ShapeEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Б
<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ы
6gradients_1/Generator/last_layer/leaky_relu_grad/zerosFill8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
я
=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
О
Fgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Generator/last_layer/leaky_relu_grad/Shape8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≤
7gradients_1/Generator/last_layer/leaky_relu_grad/SelectSelect=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency6gradients_1/Generator/last_layer/leaky_relu_grad/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
і
9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Select=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqual6gradients_1/Generator/last_layer/leaky_relu_grad/zerosEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
ь
4gradients_1/Generator/last_layer/leaky_relu_grad/SumSum7gradients_1/Generator/last_layer/leaky_relu_grad/SelectFgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
т
8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Generator/last_layer/leaky_relu_grad/Sum6gradients_1/Generator/last_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
В
6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Hgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_18gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
Ѕ
Agradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape;^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1
”
Igradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeB^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
ў
Kgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1B^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
}
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
і
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Ъ
Jgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ч
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulMulIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:€€€€€€€€€А*
T0
Е
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/SumSum8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulJgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
м
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ж
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Mul%Generator/last_layer/leaky_relu/alphaIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
Л
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Д
>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ќ
Egradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
—
Mgradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
й
Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*Q
_classG
ECloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
√
gradients_1/AddN_3AddNKgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
«
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Generator/last_layer/batch_normalization/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
Ь
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:А*
dtype0
ў
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Й
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_3_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
љ
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Н
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_3agradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ґ
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:А
М
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
Ј
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape
∞
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:А
ї
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Generator/last_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ь
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
_output_shapes
:*
valueB:А*
dtype0
ў
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
£
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Generator/last_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:€€€€€€€€€А*
T0
ƒ
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
љ
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ы
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Generator/last_layer/fully_connected/BiasAddbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
 
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ґ
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:А*
T0*
Tshape0
М
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
Ј
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
∞
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:А
ё
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
Х
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg
ї
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:А*
T0
Ь
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:А
щ
Igradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:А
З
Ngradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad
∞
Vgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
Д
Xgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
Ц
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Generator/last_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:А*
T0
Ы
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Generator/last_layer/batch_normalization/moving_mean/read*
_output_shapes	
:А*
T0
Д
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Ґ
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:А*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul
®
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:А
∞
Cgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
Ъ
Egradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/third_layer/leaky_reluVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
г
Mgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1
Б
Ugradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
€
Wgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*X
_classN
LJloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1
э
gradients_1/AddN_4AddNdgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*
_output_shapes	
:А*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
Ѕ
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_43Generator/last_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:А
»
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_48Generator/last_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:А
ю
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Ъ
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:А*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul
†
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:А*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Ы
7gradients_1/Generator/third_layer/leaky_relu_grad/ShapeShape$Generator/third_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
≤
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
ќ
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
В
=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ю
7gradients_1/Generator/third_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
в
>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
С
Ggradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/third_layer/leaky_relu_grad/Shape9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≈
8gradients_1/Generator/third_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/third_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
«
:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/third_layer/leaky_relu_grad/zerosUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
€
5gradients_1/Generator/third_layer/leaky_relu_grad/SumSum8gradients_1/Generator/third_layer/leaky_relu_grad/SelectGgradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
х
9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/third_layer/leaky_relu_grad/Sum7gradients_1/Generator/third_layer/leaky_relu_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
Е
7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Igradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ы
;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_19gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
ƒ
Bgradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
„
Jgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*L
_classB
@>loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape
Ё
Lgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
~
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
ґ
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Э
Kgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ъ
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
И
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
п
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
й
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/third_layer/leaky_relu/alphaJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
О
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
З
?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
–
Fgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
’
Ngradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
н
Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*R
_classH
FDloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А*
T0
∆
gradients_1/AddN_5AddNLgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
…
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape9Generator/third_layer/batch_normalization/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
Э
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:А*
dtype0*
_output_shapes
:
№
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Л
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_5`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ј
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
П
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_5bgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
є
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:А
П
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
ї
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
і
egradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:А
љ
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape-Generator/third_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Э
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:А
№
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
¶
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:€€€€€€€€€А
«
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumSumNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ј
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ю
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul-Generator/third_layer/fully_connected/BiasAddcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
Ќ
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1bgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:А
П
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
ї
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
і
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:А
а
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegNegegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
Ш
Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpf^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1M^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg
њ
agradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:А
†
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:А*
T0
ы
Jgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradcgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
_output_shapes	
:А*
T0*
data_formatNHWC
К
Ogradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpd^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyK^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad
і
Wgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitycgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
И
Ygradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Щ
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_17Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
Ю
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1:Generator/third_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:А
З
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulQ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
¶
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:А
ђ
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:А
≥
Dgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/third_layer/fully_connected/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(*
T0
Э
Fgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1MatMul!Generator/second_layer/leaky_reluWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
ж
Ngradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1
Е
Vgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*W
_classM
KIloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul
Г
Xgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
А
gradients_1/AddN_6AddNegradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:А*
T0
√
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_64Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:А
 
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_69Generator/third_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:А*
T0
Б
Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpM^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Ю
agradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:А*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul
§
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:А*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Э
8gradients_1/Generator/second_layer/leaky_relu_grad/ShapeShape%Generator/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
і
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
–
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2ShapeVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
Г
>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
8gradients_1/Generator/second_layer/leaky_relu_grad/zerosFill:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
е
?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
Ф
Hgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/Generator/second_layer/leaky_relu_grad/Shape:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
…
9gradients_1/Generator/second_layer/leaky_relu_grad/SelectSelect?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency8gradients_1/Generator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
Ћ
;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Select?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqual8gradients_1/Generator/second_layer/leaky_relu_grad/zerosVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
В
6gradients_1/Generator/second_layer/leaky_relu_grad/SumSum9gradients_1/Generator/second_layer/leaky_relu_grad/SelectHgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ш
:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeReshape6gradients_1/Generator/second_layer/leaky_relu_grad/Sum8gradients_1/Generator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
И
8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1Sum;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Jgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ю
<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1Reshape8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Cgradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_depsNoOp;^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape=^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1
џ
Kgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeD^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1D^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Є
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
†
Lgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
э
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulMulKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
Л
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/SumSum:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulLgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
т
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeReshape:gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
м
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Mul'Generator/second_layer/leaky_relu/alphaKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
С
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Ngradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
К
@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
”
Ggradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp?^gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeA^gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
ў
Ogradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeH^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*Q
_classG
ECloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape
с
Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1H^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*S
_classI
GEloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
…
gradients_1/AddN_7AddNMgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
Ћ
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape:Generator/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Ю
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:А*
dtype0*
_output_shapes
:
я
agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Н
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_7agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
√
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
С
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_7cgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Љ
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:А*
T0*
Tshape0
Т
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
њ
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
Є
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:А
њ
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape.Generator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ю
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:А*
dtype0*
_output_shapes
:
я
agradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
©
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:€€€€€€€€€А
 
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mulagradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
√
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
°
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul.Generator/second_layer/fully_connected/BiasAdddgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
–
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1cgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Љ
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:А*
T0*
Tshape0
Т
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
њ
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
Є
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:А
в
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/NegNegfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
Ы
Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpg^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1N^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
√
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:А*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
§
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:А
э
Kgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGraddgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:А
Н
Pgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpe^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyL^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Є
Xgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitydgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape
М
Zgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*^
_classT
RPloc:@gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ь
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_18Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
°
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Muldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1;Generator/second_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:А
К
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulR^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
™
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:А
∞
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:А
ґ
Egradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulMatMulXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency2Generator/second_layer/fully_connected/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(*
T0
Ю
Ggradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/first_layer/leaky_reluXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
й
Ogradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpF^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulH^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1
Й
Wgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityEgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulP^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
З
Ygradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityGgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1P^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*Z
_classP
NLloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1
Г
gradients_1/AddN_8AddNfgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:А
≈
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_85Generator/second_layer/batch_normalization/gamma/read*
_output_shapes	
:А*
T0
ћ
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_8:Generator/second_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:А*
T0
Д
Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpN^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Ґ
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:А
®
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:А
Ы
7gradients_1/Generator/first_layer/leaky_relu_grad/ShapeShape$Generator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
¶
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
–
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2ShapeWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
В
=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ю
7gradients_1/Generator/first_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
÷
>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
С
Ggradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/first_layer/leaky_relu_grad/Shape9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
«
8gradients_1/Generator/first_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
…
:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/first_layer/leaky_relu_grad/zerosWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
€
5gradients_1/Generator/first_layer/leaky_relu_grad/SumSum8gradients_1/Generator/first_layer/leaky_relu_grad/SelectGgradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
х
9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/first_layer/leaky_relu_grad/Sum7gradients_1/Generator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Е
7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Igradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ы
;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_19gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ƒ
Bgradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
„
Jgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
Ё
Lgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
~
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
™
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Э
Kgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
о
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
И
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
п
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
й
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/first_layer/leaky_relu/alphaJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
О
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
–
Fgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
’
Ngradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*P
_classF
DBloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape
н
Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
∆
gradients_1/AddN_9AddNLgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
™
Jgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:А
є
Ogradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_9K^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
ћ
Wgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_9P^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
И
Ygradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
≤
Dgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/first_layer/fully_connected/kernel/read*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(*
T0
Н
Fgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise_inWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	dА*
transpose_a(*
transpose_b( *
T0
ж
Ngradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1
Д
Vgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€d
В
Xgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	dА
О
beta1_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Я
beta1_power_1
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
¬
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias
|
beta1_power_1/readIdentitybeta1_power_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
О
beta2_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
Я
beta2_power_1
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
¬
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
|
beta2_power_1/readIdentitybeta2_power_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
е
SGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   А   *
dtype0*
_output_shapes
:
ѕ
IGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
и
CGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	dА
и
1Generator/first_layer/fully_connected/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	dА*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	dА
ќ
8Generator/first_layer/fully_connected/kernel/Adam/AssignAssign1Generator/first_layer/fully_connected/kernel/AdamCGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
_output_shapes
:	dА*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(
а
6Generator/first_layer/fully_connected/kernel/Adam/readIdentity1Generator/first_layer/fully_connected/kernel/Adam*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	dА*
T0
з
UGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   А   
—
KGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
о
EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	dА
к
3Generator/first_layer/fully_connected/kernel/Adam_1
VariableV2*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	dА*
dtype0*
_output_shapes
:	dА*
shared_name 
‘
:Generator/first_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/first_layer/fully_connected/kernel/Adam_1EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	dА*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(
д
8Generator/first_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/first_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	dА
ѕ
AGenerator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
№
/Generator/first_layer/fully_connected/bias/Adam
VariableV2*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
¬
6Generator/first_layer/fully_connected/bias/Adam/AssignAssign/Generator/first_layer/fully_connected/bias/AdamAGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
÷
4Generator/first_layer/fully_connected/bias/Adam/readIdentity/Generator/first_layer/fully_connected/bias/Adam*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:А*
T0
—
CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ё
1Generator/first_layer/fully_connected/bias/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container 
»
8Generator/first_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/first_layer/fully_connected/bias/Adam_1CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
Џ
6Generator/first_layer/fully_connected/bias/Adam_1/readIdentity1Generator/first_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:А
з
TGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"А      *
dtype0
—
JGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
н
DGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillTGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorJGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА
м
2Generator/second_layer/fully_connected/kernel/Adam
VariableV2*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
”
9Generator/second_layer/fully_connected/kernel/Adam/AssignAssign2Generator/second_layer/fully_connected/kernel/AdamDGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0
д
7Generator/second_layer/fully_connected/kernel/Adam/readIdentity2Generator/second_layer/fully_connected/kernel/Adam*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
й
VGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"А      
”
LGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0
у
FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillVGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА*
T0
о
4Generator/second_layer/fully_connected/kernel/Adam_1
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
ў
;Generator/second_layer/fully_connected/kernel/Adam_1/AssignAssign4Generator/second_layer/fully_connected/kernel/Adam_1FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
и
9Generator/second_layer/fully_connected/kernel/Adam_1/readIdentity4Generator/second_layer/fully_connected/kernel/Adam_1*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
—
BGenerator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ё
0Generator/second_layer/fully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:А
∆
7Generator/second_layer/fully_connected/bias/Adam/AssignAssign0Generator/second_layer/fully_connected/bias/AdamBGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
ў
5Generator/second_layer/fully_connected/bias/Adam/readIdentity0Generator/second_layer/fully_connected/bias/Adam*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:А
”
DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:А*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueBА*    *
dtype0
а
2Generator/second_layer/fully_connected/bias/Adam_1
VariableV2*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
ћ
9Generator/second_layer/fully_connected/bias/Adam_1/AssignAssign2Generator/second_layer/fully_connected/bias/Adam_1DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias
Ё
7Generator/second_layer/fully_connected/bias/Adam_1/readIdentity2Generator/second_layer/fully_connected/bias/Adam_1*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:А
џ
GGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueBА*    *
dtype0*
_output_shapes	
:А
и
5Generator/second_layer/batch_normalization/gamma/Adam
VariableV2*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
Џ
<Generator/second_layer/batch_normalization/gamma/Adam/AssignAssign5Generator/second_layer/batch_normalization/gamma/AdamGGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
и
:Generator/second_layer/batch_normalization/gamma/Adam/readIdentity5Generator/second_layer/batch_normalization/gamma/Adam*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:А*
T0
Ё
IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueBА*    
к
7Generator/second_layer/batch_normalization/gamma/Adam_1
VariableV2*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:А*
dtype0*
_output_shapes	
:А
а
>Generator/second_layer/batch_normalization/gamma/Adam_1/AssignAssign7Generator/second_layer/batch_normalization/gamma/Adam_1IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А*
use_locking(
м
<Generator/second_layer/batch_normalization/gamma/Adam_1/readIdentity7Generator/second_layer/batch_normalization/gamma/Adam_1*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:А
ў
FGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
ж
4Generator/second_layer/batch_normalization/beta/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container 
÷
;Generator/second_layer/batch_normalization/beta/Adam/AssignAssign4Generator/second_layer/batch_normalization/beta/AdamFGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
е
9Generator/second_layer/batch_normalization/beta/Adam/readIdentity4Generator/second_layer/batch_normalization/beta/Adam*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:А
џ
HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*
_output_shapes	
:А*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueBА*    *
dtype0
и
6Generator/second_layer/batch_normalization/beta/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:А
№
=Generator/second_layer/batch_normalization/beta/Adam_1/AssignAssign6Generator/second_layer/batch_normalization/beta/Adam_1HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
й
;Generator/second_layer/batch_normalization/beta/Adam_1/readIdentity6Generator/second_layer/batch_normalization/beta/Adam_1*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:А
е
SGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ѕ
IGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
й
CGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0
к
1Generator/third_layer/fully_connected/kernel/Adam
VariableV2*
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container 
ѕ
8Generator/third_layer/fully_connected/kernel/Adam/AssignAssign1Generator/third_layer/fully_connected/kernel/AdamCGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
б
6Generator/third_layer/fully_connected/kernel/Adam/readIdentity1Generator/third_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
АА
з
UGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      
—
KGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0
п
EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА
м
3Generator/third_layer/fully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
АА
’
:Generator/third_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/third_layer/fully_connected/kernel/Adam_1EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0
е
8Generator/third_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/third_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
АА
ѕ
AGenerator/third_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
№
/Generator/third_layer/fully_connected/bias/Adam
VariableV2*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
¬
6Generator/third_layer/fully_connected/bias/Adam/AssignAssign/Generator/third_layer/fully_connected/bias/AdamAGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
÷
4Generator/third_layer/fully_connected/bias/Adam/readIdentity/Generator/third_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:А
—
CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueBА*    
ё
1Generator/third_layer/fully_connected/bias/Adam_1
VariableV2*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
»
8Generator/third_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/third_layer/fully_connected/bias/Adam_1CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Џ
6Generator/third_layer/fully_connected/bias/Adam_1/readIdentity1Generator/third_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:А
ў
FGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueBА*    *
dtype0*
_output_shapes	
:А
ж
4Generator/third_layer/batch_normalization/gamma/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container 
÷
;Generator/third_layer/batch_normalization/gamma/Adam/AssignAssign4Generator/third_layer/batch_normalization/gamma/AdamFGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
е
9Generator/third_layer/batch_normalization/gamma/Adam/readIdentity4Generator/third_layer/batch_normalization/gamma/Adam*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:А
џ
HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueBА*    *
dtype0*
_output_shapes	
:А
и
6Generator/third_layer/batch_normalization/gamma/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container 
№
=Generator/third_layer/batch_normalization/gamma/Adam_1/AssignAssign6Generator/third_layer/batch_normalization/gamma/Adam_1HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
й
;Generator/third_layer/batch_normalization/gamma/Adam_1/readIdentity6Generator/third_layer/batch_normalization/gamma/Adam_1*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:А
„
EGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueBА*    
д
3Generator/third_layer/batch_normalization/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:А
“
:Generator/third_layer/batch_normalization/beta/Adam/AssignAssign3Generator/third_layer/batch_normalization/beta/AdamEGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
в
8Generator/third_layer/batch_normalization/beta/Adam/readIdentity3Generator/third_layer/batch_normalization/beta/Adam*
_output_shapes	
:А*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
ў
GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*
_output_shapes	
:А*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueBА*    *
dtype0
ж
5Generator/third_layer/batch_normalization/beta/Adam_1
VariableV2*
_output_shapes	
:А*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:А*
dtype0
Ў
<Generator/third_layer/batch_normalization/beta/Adam_1/AssignAssign5Generator/third_layer/batch_normalization/beta/Adam_1GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(
ж
:Generator/third_layer/batch_normalization/beta/Adam_1/readIdentity5Generator/third_layer/batch_normalization/beta/Adam_1*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:А
г
RGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      
Ќ
HGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
е
BGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zerosFillRGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА
и
0Generator/last_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
АА
Ћ
7Generator/last_layer/fully_connected/kernel/Adam/AssignAssign0Generator/last_layer/fully_connected/kernel/AdamBGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0
ё
5Generator/last_layer/fully_connected/kernel/Adam/readIdentity0Generator/last_layer/fully_connected/kernel/Adam*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
АА
е
TGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      
ѕ
JGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
л
DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillTGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorJGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА
к
2Generator/last_layer/fully_connected/kernel/Adam_1
VariableV2*
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container 
—
9Generator/last_layer/fully_connected/kernel/Adam_1/AssignAssign2Generator/last_layer/fully_connected/kernel/Adam_1DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
в
7Generator/last_layer/fully_connected/kernel/Adam_1/readIdentity2Generator/last_layer/fully_connected/kernel/Adam_1*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
АА
ў
PGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:А
…
FGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Ў
@Generator/last_layer/fully_connected/bias/Adam/Initializer/zerosFillPGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorFGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:А
Џ
.Generator/last_layer/fully_connected/bias/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container 
Њ
5Generator/last_layer/fully_connected/bias/Adam/AssignAssign.Generator/last_layer/fully_connected/bias/Adam@Generator/last_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
”
3Generator/last_layer/fully_connected/bias/Adam/readIdentity.Generator/last_layer/fully_connected/bias/Adam*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:А
џ
RGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:А*
dtype0*
_output_shapes
:
Ћ
HGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
ё
BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zerosFillRGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:А
№
0Generator/last_layer/fully_connected/bias/Adam_1
VariableV2*
_output_shapes	
:А*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:А*
dtype0
ƒ
7Generator/last_layer/fully_connected/bias/Adam_1/AssignAssign0Generator/last_layer/fully_connected/bias/Adam_1BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
„
5Generator/last_layer/fully_connected/bias/Adam_1/readIdentity0Generator/last_layer/fully_connected/bias/Adam_1*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:А
г
UGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:А*
dtype0*
_output_shapes
:
”
KGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0
м
EGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zerosFillUGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorKGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/Const*
_output_shapes	
:А*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
д
3Generator/last_layer/batch_normalization/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:А
“
:Generator/last_layer/batch_normalization/gamma/Adam/AssignAssign3Generator/last_layer/batch_normalization/gamma/AdamEGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
в
8Generator/last_layer/batch_normalization/gamma/Adam/readIdentity3Generator/last_layer/batch_normalization/gamma/Adam*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:А
е
WGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:А*
dtype0*
_output_shapes
:
’
MGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    
т
GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zerosFillWGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorMGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/Const*
_output_shapes	
:А*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
ж
5Generator/last_layer/batch_normalization/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:А
Ў
<Generator/last_layer/batch_normalization/gamma/Adam_1/AssignAssign5Generator/last_layer/batch_normalization/gamma/Adam_1GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
ж
:Generator/last_layer/batch_normalization/gamma/Adam_1/readIdentity5Generator/last_layer/batch_normalization/gamma/Adam_1*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:А
б
TGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:А*
dtype0*
_output_shapes
:
—
JGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
и
DGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zerosFillTGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorJGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:А
в
2Generator/last_layer/batch_normalization/beta/Adam
VariableV2*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А
ќ
9Generator/last_layer/batch_normalization/beta/Adam/AssignAssign2Generator/last_layer/batch_normalization/beta/AdamDGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
я
7Generator/last_layer/batch_normalization/beta/Adam/readIdentity2Generator/last_layer/batch_normalization/beta/Adam*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:А
г
VGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:А*
dtype0*
_output_shapes
:
”
LGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0
о
FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zerosFillVGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:А
д
4Generator/last_layer/batch_normalization/beta/Adam_1
VariableV2*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
‘
;Generator/last_layer/batch_normalization/beta/Adam_1/AssignAssign4Generator/last_layer/batch_normalization/beta/Adam_1FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
г
9Generator/last_layer/batch_normalization/beta/Adam_1/readIdentity4Generator/last_layer/batch_normalization/beta/Adam_1*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:А*
T0
√
BGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
≠
8Generator/fake_image/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0
•
2Generator/fake_image/kernel/Adam/Initializer/zerosFillBGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensor8Generator/fake_image/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:
АР
»
 Generator/fake_image/kernel/Adam
VariableV2* 
_output_shapes
:
АР*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
АР*
dtype0
Л
'Generator/fake_image/kernel/Adam/AssignAssign Generator/fake_image/kernel/Adam2Generator/fake_image/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
АР
Ѓ
%Generator/fake_image/kernel/Adam/readIdentity Generator/fake_image/kernel/Adam*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
АР*
T0
≈
DGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
ѓ
:Generator/fake_image/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ђ
4Generator/fake_image/kernel/Adam_1/Initializer/zerosFillDGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensor:Generator/fake_image/kernel/Adam_1/Initializer/zeros/Const*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:
АР*
T0
 
"Generator/fake_image/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
АР*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
АР
С
)Generator/fake_image/kernel/Adam_1/AssignAssign"Generator/fake_image/kernel/Adam_14Generator/fake_image/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
АР*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(
≤
'Generator/fake_image/kernel/Adam_1/readIdentity"Generator/fake_image/kernel/Adam_1*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
АР
≠
0Generator/fake_image/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueBР*    *
dtype0*
_output_shapes	
:Р
Ї
Generator/fake_image/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:Р*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:Р
ю
%Generator/fake_image/bias/Adam/AssignAssignGenerator/fake_image/bias/Adam0Generator/fake_image/bias/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:Р
£
#Generator/fake_image/bias/Adam/readIdentityGenerator/fake_image/bias/Adam*
_output_shapes	
:Р*
T0*,
_class"
 loc:@Generator/fake_image/bias
ѓ
2Generator/fake_image/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueBР*    *
dtype0*
_output_shapes	
:Р
Љ
 Generator/fake_image/bias/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:Р*
dtype0*
_output_shapes	
:Р
Д
'Generator/fake_image/bias/Adam_1/AssignAssign Generator/fake_image/bias/Adam_12Generator/fake_image/bias/Adam_1/Initializer/zeros*
_output_shapes	
:Р*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
І
%Generator/fake_image/bias/Adam_1/readIdentity Generator/fake_image/bias/Adam_1*
_output_shapes	
:Р*
T0*,
_class"
 loc:@Generator/fake_image/bias
Y
Adam_1/learning_rateConst*
_output_shapes
: *
valueB
 *ЈQ9*
dtype0
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
_output_shapes
: *
valueB
 *wЊ?*
dtype0
S
Adam_1/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
љ
DAdam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/first_layer/fully_connected/kernel1Generator/first_layer/fully_connected/kernel/Adam3Generator/first_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	dА
∞
BAdam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/first_layer/fully_connected/bias/Generator/first_layer/fully_connected/bias/Adam1Generator/first_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:А*
use_locking( *
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
use_nesterov( 
ƒ
EAdam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam-Generator/second_layer/fully_connected/kernel2Generator/second_layer/fully_connected/kernel/Adam4Generator/second_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
АА
ґ
CAdam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam+Generator/second_layer/fully_connected/bias0Generator/second_layer/fully_connected/bias/Adam2Generator/second_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0
ў
HAdam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam0Generator/second_layer/batch_normalization/gamma5Generator/second_layer/batch_normalization/gamma/Adam7Generator/second_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:А*
use_locking( *
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
use_nesterov( 
“
GAdam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam/Generator/second_layer/batch_normalization/beta4Generator/second_layer/batch_normalization/beta/Adam6Generator/second_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:А
Њ
DAdam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/third_layer/fully_connected/kernel1Generator/third_layer/fully_connected/kernel/Adam3Generator/third_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
АА
∞
BAdam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/third_layer/fully_connected/bias/Generator/third_layer/fully_connected/bias/Adam1Generator/third_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:А*
use_locking( *
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
use_nesterov( 
”
GAdam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam/Generator/third_layer/batch_normalization/gamma4Generator/third_layer/batch_normalization/gamma/Adam6Generator/third_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:А*
use_locking( *
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
use_nesterov( 
ћ
FAdam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdam	ApplyAdam.Generator/third_layer/batch_normalization/beta3Generator/third_layer/batch_normalization/beta/Adam5Generator/third_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0
Є
CAdam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Generator/last_layer/fully_connected/kernel0Generator/last_layer/fully_connected/kernel/Adam2Generator/last_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
АА*
use_locking( *
T0
™
AAdam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Generator/last_layer/fully_connected/bias.Generator/last_layer/fully_connected/bias/Adam0Generator/last_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:А*
use_locking( *
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
use_nesterov( 
Ќ
FAdam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Generator/last_layer/batch_normalization/gamma3Generator/last_layer/batch_normalization/gamma/Adam5Generator/last_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:А*
use_locking( *
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
use_nesterov( 
∆
EAdam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Generator/last_layer/batch_normalization/beta2Generator/last_layer/batch_normalization/beta/Adam4Generator/last_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0
Ў
3Adam_1/update_Generator/fake_image/kernel/ApplyAdam	ApplyAdamGenerator/fake_image/kernel Generator/fake_image/kernel/Adam"Generator/fake_image/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
АР*
use_locking( *
T0*.
_class$
" loc:@Generator/fake_image/kernel*
use_nesterov( 
 
1Adam_1/update_Generator/fake_image/bias/ApplyAdam	ApplyAdamGenerator/fake_image/biasGenerator/fake_image/bias/Adam Generator/fake_image/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:Р*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
use_nesterov( 
’	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
™
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
„	
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta22^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
Ѓ
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
Г	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: "/AXх÷l     ФіCЯ	ЭҐhdкэ÷AJ…ў
яљ
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
2	АР
о
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
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

2	Р
Н
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
2	Р
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
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
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
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype*1.12.02v1.12.0-0-ga6d8ffae09цТ
u
Generator/noise_inPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€d*
shape:€€€€€€€€€d
я
MGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   А   *
dtype0*
_output_shapes
:
—
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&Њ*
dtype0*
_output_shapes
: 
—
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&>
∆
UGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	dА*

seed *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
seed2 
ќ
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
: 
б
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
_output_shapes
:	dА*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
”
GGenerator/first_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
:	dА*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
г
,Generator/first_layer/fully_connected/kernel
VariableV2*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	dА*
dtype0*
_output_shapes
:	dА*
shared_name 
»
3Generator/first_layer/fully_connected/kernel/AssignAssign,Generator/first_layer/fully_connected/kernelGGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	dА
÷
1Generator/first_layer/fully_connected/kernel/readIdentity,Generator/first_layer/fully_connected/kernel*
_output_shapes
:	dА*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
 
<Generator/first_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
„
*Generator/first_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:А
≥
1Generator/first_layer/fully_connected/bias/AssignAssign*Generator/first_layer/fully_connected/bias<Generator/first_layer/fully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
ћ
/Generator/first_layer/fully_connected/bias/readIdentity*Generator/first_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:А
÷
,Generator/first_layer/fully_connected/MatMulMatMulGenerator/noise_in1Generator/first_layer/fully_connected/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
б
-Generator/first_layer/fully_connected/BiasAddBiasAdd,Generator/first_layer/fully_connected/MatMul/Generator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
k
&Generator/first_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
µ
$Generator/first_layer/leaky_relu/mulMul&Generator/first_layer/leaky_relu/alpha-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
≥
 Generator/first_layer/leaky_reluMaximum$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
б
NGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"А      *
dtype0*
_output_shapes
:
”
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   Њ*
dtype0*
_output_shapes
: 
”
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
 
VGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformNGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
“
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
ж
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulVGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
Ў
HGenerator/second_layer/fully_connected/kernel/Initializer/random_uniformAddLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
з
-Generator/second_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
АА
Ќ
4Generator/second_layer/fully_connected/kernel/AssignAssign-Generator/second_layer/fully_connected/kernelHGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА*
use_locking(
Џ
2Generator/second_layer/fully_connected/kernel/readIdentity-Generator/second_layer/fully_connected/kernel*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
ћ
=Generator/second_layer/fully_connected/bias/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ў
+Generator/second_layer/fully_connected/bias
VariableV2*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
Ј
2Generator/second_layer/fully_connected/bias/AssignAssign+Generator/second_layer/fully_connected/bias=Generator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
ѕ
0Generator/second_layer/fully_connected/bias/readIdentity+Generator/second_layer/fully_connected/bias*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:А
ж
-Generator/second_layer/fully_connected/MatMulMatMul Generator/first_layer/leaky_relu2Generator/second_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
д
.Generator/second_layer/fully_connected/BiasAddBiasAdd-Generator/second_layer/fully_connected/MatMul0Generator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
’
AGenerator/second_layer/batch_normalization/gamma/Initializer/onesConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueBА*  А?*
dtype0*
_output_shapes	
:А
г
0Generator/second_layer/batch_normalization/gamma
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
 
7Generator/second_layer/batch_normalization/gamma/AssignAssign0Generator/second_layer/batch_normalization/gammaAGenerator/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
ё
5Generator/second_layer/batch_normalization/gamma/readIdentity0Generator/second_layer/batch_normalization/gamma*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:А
‘
AGenerator/second_layer/batch_normalization/beta/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
б
/Generator/second_layer/batch_normalization/beta
VariableV2*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А
«
6Generator/second_layer/batch_normalization/beta/AssignAssign/Generator/second_layer/batch_normalization/betaAGenerator/second_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
џ
4Generator/second_layer/batch_normalization/beta/readIdentity/Generator/second_layer/batch_normalization/beta*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:А
в
HGenerator/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
valueBА*    
п
6Generator/second_layer/batch_normalization/moving_mean
VariableV2*
shared_name *I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
	container *
shape:А*
dtype0*
_output_shapes	
:А
г
=Generator/second_layer/batch_normalization/moving_mean/AssignAssign6Generator/second_layer/batch_normalization/moving_meanHGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:А*
use_locking(
р
;Generator/second_layer/batch_normalization/moving_mean/readIdentity6Generator/second_layer/batch_normalization/moving_mean*
_output_shapes	
:А*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean
й
KGenerator/second_layer/batch_normalization/moving_variance/Initializer/onesConst*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
valueBА*  А?*
dtype0*
_output_shapes	
:А
ч
:Generator/second_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
	container *
shape:А
т
AGenerator/second_layer/batch_normalization/moving_variance/AssignAssign:Generator/second_layer/batch_normalization/moving_varianceKGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:А
ь
?Generator/second_layer/batch_normalization/moving_variance/readIdentity:Generator/second_layer/batch_normalization/moving_variance*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
_output_shapes	
:А

:Generator/second_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
в
8Generator/second_layer/batch_normalization/batchnorm/addAdd?Generator/second_layer/batch_normalization/moving_variance/read:Generator/second_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:А*
T0
£
:Generator/second_layer/batch_normalization/batchnorm/RsqrtRsqrt8Generator/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:А
Ў
8Generator/second_layer/batch_normalization/batchnorm/mulMul:Generator/second_layer/batch_normalization/batchnorm/Rsqrt5Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:А
ё
:Generator/second_layer/batch_normalization/batchnorm/mul_1Mul.Generator/second_layer/fully_connected/BiasAdd8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:€€€€€€€€€А
ё
:Generator/second_layer/batch_normalization/batchnorm/mul_2Mul;Generator/second_layer/batch_normalization/moving_mean/read8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
„
8Generator/second_layer/batch_normalization/batchnorm/subSub4Generator/second_layer/batch_normalization/beta/read:Generator/second_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:А*
T0
к
:Generator/second_layer/batch_normalization/batchnorm/add_1Add:Generator/second_layer/batch_normalization/batchnorm/mul_18Generator/second_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:€€€€€€€€€А
l
'Generator/second_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
ƒ
%Generator/second_layer/leaky_relu/mulMul'Generator/second_layer/leaky_relu/alpha:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:€€€€€€€€€А*
T0
¬
!Generator/second_layer/leaky_reluMaximum%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:€€€€€€€€€А*
T0
я
MGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      
—
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *уµљ*
dtype0*
_output_shapes
: 
—
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *уµ=*
dtype0*
_output_shapes
: 
«
UGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shape*

seed *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА
ќ
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
_output_shapes
: 
в
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
АА
‘
GGenerator/third_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
АА
е
,Generator/third_layer/fully_connected/kernel
VariableV2*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
…
3Generator/third_layer/fully_connected/kernel/AssignAssign,Generator/third_layer/fully_connected/kernelGGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
„
1Generator/third_layer/fully_connected/kernel/readIdentity,Generator/third_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
АА
 
<Generator/third_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
„
*Generator/third_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:А
≥
1Generator/third_layer/fully_connected/bias/AssignAssign*Generator/third_layer/fully_connected/bias<Generator/third_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
ћ
/Generator/third_layer/fully_connected/bias/readIdentity*Generator/third_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:А
е
,Generator/third_layer/fully_connected/MatMulMatMul!Generator/second_layer/leaky_relu1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
б
-Generator/third_layer/fully_connected/BiasAddBiasAdd,Generator/third_layer/fully_connected/MatMul/Generator/third_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
”
@Generator/third_layer/batch_normalization/gamma/Initializer/onesConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueBА*  А?*
dtype0*
_output_shapes	
:А
б
/Generator/third_layer/batch_normalization/gamma
VariableV2*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
∆
6Generator/third_layer/batch_normalization/gamma/AssignAssign/Generator/third_layer/batch_normalization/gamma@Generator/third_layer/batch_normalization/gamma/Initializer/ones*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А*
use_locking(
џ
4Generator/third_layer/batch_normalization/gamma/readIdentity/Generator/third_layer/batch_normalization/gamma*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:А
“
@Generator/third_layer/batch_normalization/beta/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
я
.Generator/third_layer/batch_normalization/beta
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
√
5Generator/third_layer/batch_normalization/beta/AssignAssign.Generator/third_layer/batch_normalization/beta@Generator/third_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
Ў
3Generator/third_layer/batch_normalization/beta/readIdentity.Generator/third_layer/batch_normalization/beta*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:А
а
GGenerator/third_layer/batch_normalization/moving_mean/Initializer/zerosConst*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
valueBА*    *
dtype0*
_output_shapes	
:А
н
5Generator/third_layer/batch_normalization/moving_mean
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
	container *
shape:А
я
<Generator/third_layer/batch_normalization/moving_mean/AssignAssign5Generator/third_layer/batch_normalization/moving_meanGGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:А*
use_locking(
н
:Generator/third_layer/batch_normalization/moving_mean/readIdentity5Generator/third_layer/batch_normalization/moving_mean*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
_output_shapes	
:А
з
JGenerator/third_layer/batch_normalization/moving_variance/Initializer/onesConst*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
valueBА*  А?*
dtype0*
_output_shapes	
:А
х
9Generator/third_layer/batch_normalization/moving_variance
VariableV2*
shared_name *L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
	container *
shape:А*
dtype0*
_output_shapes	
:А
о
@Generator/third_layer/batch_normalization/moving_variance/AssignAssign9Generator/third_layer/batch_normalization/moving_varianceJGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:А
щ
>Generator/third_layer/batch_normalization/moving_variance/readIdentity9Generator/third_layer/batch_normalization/moving_variance*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
_output_shapes	
:А
~
9Generator/third_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
я
7Generator/third_layer/batch_normalization/batchnorm/addAdd>Generator/third_layer/batch_normalization/moving_variance/read9Generator/third_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:А*
T0
°
9Generator/third_layer/batch_normalization/batchnorm/RsqrtRsqrt7Generator/third_layer/batch_normalization/batchnorm/add*
_output_shapes	
:А*
T0
’
7Generator/third_layer/batch_normalization/batchnorm/mulMul9Generator/third_layer/batch_normalization/batchnorm/Rsqrt4Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:А
џ
9Generator/third_layer/batch_normalization/batchnorm/mul_1Mul-Generator/third_layer/fully_connected/BiasAdd7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:€€€€€€€€€А
џ
9Generator/third_layer/batch_normalization/batchnorm/mul_2Mul:Generator/third_layer/batch_normalization/moving_mean/read7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
‘
7Generator/third_layer/batch_normalization/batchnorm/subSub3Generator/third_layer/batch_normalization/beta/read9Generator/third_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:А
з
9Generator/third_layer/batch_normalization/batchnorm/add_1Add9Generator/third_layer/batch_normalization/batchnorm/mul_17Generator/third_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:€€€€€€€€€А
k
&Generator/third_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
Ѕ
$Generator/third_layer/leaky_relu/mulMul&Generator/third_layer/leaky_relu/alpha9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
њ
 Generator/third_layer/leaky_reluMaximum$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
Ё
LGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ѕ
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  Аљ*
dtype0*
_output_shapes
: 
ѕ
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  А=
ƒ
TGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АА*

seed *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
seed2 
 
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/subSubJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
_output_shapes
: 
ё
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
АА
–
FGenerator/last_layer/fully_connected/kernel/Initializer/random_uniformAddJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
г
+Generator/last_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
АА
≈
2Generator/last_layer/fully_connected/kernel/AssignAssign+Generator/last_layer/fully_connected/kernelFGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
‘
0Generator/last_layer/fully_connected/kernel/readIdentity+Generator/last_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
АА
‘
KGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:А*
dtype0*
_output_shapes
:
ƒ
AGenerator/last_layer/fully_connected/bias/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
…
;Generator/last_layer/fully_connected/bias/Initializer/zerosFillKGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorAGenerator/last_layer/fully_connected/bias/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:А
’
)Generator/last_layer/fully_connected/bias
VariableV2*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
ѓ
0Generator/last_layer/fully_connected/bias/AssignAssign)Generator/last_layer/fully_connected/bias;Generator/last_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
…
.Generator/last_layer/fully_connected/bias/readIdentity)Generator/last_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:А
в
+Generator/last_layer/fully_connected/MatMulMatMul Generator/third_layer/leaky_relu0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
ё
,Generator/last_layer/fully_connected/BiasAddBiasAdd+Generator/last_layer/fully_connected/MatMul.Generator/last_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
Ё
OGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:А
Ќ
EGenerator/last_layer/batch_normalization/gamma/Initializer/ones/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Џ
?Generator/last_layer/batch_normalization/gamma/Initializer/onesFillOGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorEGenerator/last_layer/batch_normalization/gamma/Initializer/ones/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:А
я
.Generator/last_layer/batch_normalization/gamma
VariableV2*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:А*
dtype0*
_output_shapes	
:А
¬
5Generator/last_layer/batch_normalization/gamma/AssignAssign.Generator/last_layer/batch_normalization/gamma?Generator/last_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
Ў
3Generator/last_layer/batch_normalization/gamma/readIdentity.Generator/last_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:А
№
OGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:А*
dtype0*
_output_shapes
:
ћ
EGenerator/last_layer/batch_normalization/beta/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ў
?Generator/last_layer/batch_normalization/beta/Initializer/zerosFillOGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorEGenerator/last_layer/batch_normalization/beta/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:А
Ё
-Generator/last_layer/batch_normalization/beta
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
њ
4Generator/last_layer/batch_normalization/beta/AssignAssign-Generator/last_layer/batch_normalization/beta?Generator/last_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
’
2Generator/last_layer/batch_normalization/beta/readIdentity-Generator/last_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:А
к
VGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB:А*
dtype0*
_output_shapes
:
Џ
LGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
х
FGenerator/last_layer/batch_normalization/moving_mean/Initializer/zerosFillVGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*

index_type0*
_output_shapes	
:А
л
4Generator/last_layer/batch_normalization/moving_mean
VariableV2*
shared_name *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
	container *
shape:А*
dtype0*
_output_shapes	
:А
џ
;Generator/last_layer/batch_normalization/moving_mean/AssignAssign4Generator/last_layer/batch_normalization/moving_meanFGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:А
к
9Generator/last_layer/batch_normalization/moving_mean/readIdentity4Generator/last_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
_output_shapes	
:А
с
YGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB:А*
dtype0*
_output_shapes
:
б
OGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB
 *  А?
В
IGenerator/last_layer/batch_normalization/moving_variance/Initializer/onesFillYGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorOGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/Const*
_output_shapes	
:А*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*

index_type0
у
8Generator/last_layer/batch_normalization/moving_variance
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
	container 
к
?Generator/last_layer/batch_normalization/moving_variance/AssignAssign8Generator/last_layer/batch_normalization/moving_varianceIGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:А*
use_locking(
ц
=Generator/last_layer/batch_normalization/moving_variance/readIdentity8Generator/last_layer/batch_normalization/moving_variance*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
_output_shapes	
:А
}
8Generator/last_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
№
6Generator/last_layer/batch_normalization/batchnorm/addAdd=Generator/last_layer/batch_normalization/moving_variance/read8Generator/last_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:А
Я
8Generator/last_layer/batch_normalization/batchnorm/RsqrtRsqrt6Generator/last_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:А
“
6Generator/last_layer/batch_normalization/batchnorm/mulMul8Generator/last_layer/batch_normalization/batchnorm/Rsqrt3Generator/last_layer/batch_normalization/gamma/read*
_output_shapes	
:А*
T0
Ў
8Generator/last_layer/batch_normalization/batchnorm/mul_1Mul,Generator/last_layer/fully_connected/BiasAdd6Generator/last_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:€€€€€€€€€А*
T0
Ў
8Generator/last_layer/batch_normalization/batchnorm/mul_2Mul9Generator/last_layer/batch_normalization/moving_mean/read6Generator/last_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:А*
T0
—
6Generator/last_layer/batch_normalization/batchnorm/subSub2Generator/last_layer/batch_normalization/beta/read8Generator/last_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:А
д
8Generator/last_layer/batch_normalization/batchnorm/add_1Add8Generator/last_layer/batch_normalization/batchnorm/mul_16Generator/last_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:€€€€€€€€€А
j
%Generator/last_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ЌћL>
Њ
#Generator/last_layer/leaky_relu/mulMul%Generator/last_layer/leaky_relu/alpha8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
Љ
Generator/last_layer/leaky_reluMaximum#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:€€€€€€€€€А*
T0
љ
<Generator/fake_image/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
ѓ
:Generator/fake_image/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zхkљ*
dtype0*
_output_shapes
: 
ѓ
:Generator/fake_image/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zхk=
Ф
DGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniformRandomUniform<Generator/fake_image/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
АР*

seed *
T0*.
_class$
" loc:@Generator/fake_image/kernel*
seed2 
К
:Generator/fake_image/kernel/Initializer/random_uniform/subSub:Generator/fake_image/kernel/Initializer/random_uniform/max:Generator/fake_image/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*.
_class$
" loc:@Generator/fake_image/kernel
Ю
:Generator/fake_image/kernel/Initializer/random_uniform/mulMulDGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniform:Generator/fake_image/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
АР*
T0*.
_class$
" loc:@Generator/fake_image/kernel
Р
6Generator/fake_image/kernel/Initializer/random_uniformAdd:Generator/fake_image/kernel/Initializer/random_uniform/mul:Generator/fake_image/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
АР
√
Generator/fake_image/kernel
VariableV2*
dtype0* 
_output_shapes
:
АР*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
АР
Е
"Generator/fake_image/kernel/AssignAssignGenerator/fake_image/kernel6Generator/fake_image/kernel/Initializer/random_uniform*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
АР
§
 Generator/fake_image/kernel/readIdentityGenerator/fake_image/kernel*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
АР
®
+Generator/fake_image/bias/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueBР*    *
dtype0*
_output_shapes	
:Р
µ
Generator/fake_image/bias
VariableV2*
	container *
shape:Р*
dtype0*
_output_shapes	
:Р*
shared_name *,
_class"
 loc:@Generator/fake_image/bias
п
 Generator/fake_image/bias/AssignAssignGenerator/fake_image/bias+Generator/fake_image/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:Р*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias
Щ
Generator/fake_image/bias/readIdentityGenerator/fake_image/bias*
_output_shapes	
:Р*
T0*,
_class"
 loc:@Generator/fake_image/bias
Ѕ
Generator/fake_image/MatMulMatMulGenerator/last_layer/leaky_relu Generator/fake_image/kernel/read*
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€Р*
transpose_a( 
Ѓ
Generator/fake_image/BiasAddBiasAddGenerator/fake_image/MatMulGenerator/fake_image/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€Р*
T0
r
Generator/fake_image/TanhTanhGenerator/fake_image/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€Р
z
Discriminator/real_inPlaceholder*
shape:€€€€€€€€€Р*
dtype0*(
_output_shapes
:€€€€€€€€€Р
з
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ў
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HYЛљ*
dtype0*
_output_shapes
: 
ў
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HYЛ=
”
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
РА*

seed *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
seed2 
ё
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
: 
т
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
РА
д
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
РА
н
0Discriminator/first_layer/fully_connected/kernel
VariableV2*
shape:
РА*
dtype0* 
_output_shapes
:
РА*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container 
ў
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
РА*
use_locking(
г
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
РА
“
@Discriminator/first_layer/fully_connected/bias/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
я
.Discriminator/first_layer/fully_connected/bias
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
√
5Discriminator/first_layer/fully_connected/bias/AssignAssign.Discriminator/first_layer/fully_connected/bias@Discriminator/first_layer/fully_connected/bias/Initializer/zeros*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Ў
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:А*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
б
0Discriminator/first_layer/fully_connected/MatMulMatMulDiscriminator/real_in5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
н
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
o
*Discriminator/first_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
Ѕ
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
њ
$Discriminator/first_layer/leaky_reluMaximum(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
й
RDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
џ
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *уµљ*
dtype0*
_output_shapes
: 
џ
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *уµ=
÷
ZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformRDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА*

seed 
в
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
_output_shapes
: 
ц
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
и
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
АА*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
п
1Discriminator/second_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
АА
Ё
8Discriminator/second_layer/fully_connected/kernel/AssignAssign1Discriminator/second_layer/fully_connected/kernelLDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
ж
6Discriminator/second_layer/fully_connected/kernel/readIdentity1Discriminator/second_layer/fully_connected/kernel*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
‘
ADiscriminator/second_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueBА*    
б
/Discriminator/second_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:А
«
6Discriminator/second_layer/fully_connected/bias/AssignAssign/Discriminator/second_layer/fully_connected/biasADiscriminator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
џ
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:А
т
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
р
2Discriminator/second_layer/fully_connected/BiasAddBiasAdd1Discriminator/second_layer/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
p
+Discriminator/second_layer/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
ƒ
)Discriminator/second_layer/leaky_relu/mulMul+Discriminator/second_layer/leaky_relu/alpha2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
¬
%Discriminator/second_layer/leaky_reluMaximum)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
є
:Discriminator/prob/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ђ
8Discriminator/prob/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *IvЊ*
dtype0*
_output_shapes
: 
Ђ
8Discriminator/prob/kernel/Initializer/random_uniform/maxConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
Н
BDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Discriminator/prob/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А*

seed *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
seed2 
В
8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
: 
Х
8Discriminator/prob/kernel/Initializer/random_uniform/mulMulBDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniform8Discriminator/prob/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	А
З
4Discriminator/prob/kernel/Initializer/random_uniformAdd8Discriminator/prob/kernel/Initializer/random_uniform/mul8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	А
љ
Discriminator/prob/kernel
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel
ь
 Discriminator/prob/kernel/AssignAssignDiscriminator/prob/kernel4Discriminator/prob/kernel/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	А
Э
Discriminator/prob/kernel/readIdentityDiscriminator/prob/kernel*
_output_shapes
:	А*
T0*,
_class"
 loc:@Discriminator/prob/kernel
Ґ
)Discriminator/prob/bias/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
ѓ
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
ж
Discriminator/prob/bias/AssignAssignDiscriminator/prob/bias)Discriminator/prob/bias/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
Т
Discriminator/prob/bias/readIdentityDiscriminator/prob/bias*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
¬
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
І
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
з
2Discriminator/first_layer_1/fully_connected/MatMulMatMulGenerator/fake_image/Tanh5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
с
3Discriminator/first_layer_1/fully_connected/BiasAddBiasAdd2Discriminator/first_layer_1/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
q
,Discriminator/first_layer_1/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
«
*Discriminator/first_layer_1/leaky_relu/mulMul,Discriminator/first_layer_1/leaky_relu/alpha3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
≈
&Discriminator/first_layer_1/leaky_reluMaximum*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
ц
3Discriminator/second_layer_1/fully_connected/MatMulMatMul&Discriminator/first_layer_1/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
ф
4Discriminator/second_layer_1/fully_connected/BiasAddBiasAdd3Discriminator/second_layer_1/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
r
-Discriminator/second_layer_1/leaky_relu/alphaConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
 
+Discriminator/second_layer_1/leaky_relu/mulMul-Discriminator/second_layer_1/leaky_relu/alpha4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
»
'Discriminator/second_layer_1/leaky_reluMaximum+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
∆
Discriminator/prob_1/MatMulMatMul'Discriminator/second_layer_1/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Ђ
Discriminator/prob_1/BiasAddBiasAddDiscriminator/prob_1/MatMulDiscriminator/prob/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
i
ones_like/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
T
ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
w
	ones_likeFillones_like/Shapeones_like/Const*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
s
logistic_loss/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Т
logistic_loss/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:€€€€€€€€€*
T0
Ґ
logistic_loss/SelectSelectlogistic_loss/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
f
logistic_loss/NegNegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Э
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/NegDiscriminator/prob/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
q
logistic_loss/mulMulDiscriminator/prob/BiasAdd	ones_like*'
_output_shapes
:€€€€€€€€€*
T0
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0*'
_output_shapes
:€€€€€€€€€
b
logistic_loss/ExpExplogistic_loss/Select_1*
T0*'
_output_shapes
:€€€€€€€€€
a
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*'
_output_shapes
:€€€€€€€€€
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0*'
_output_shapes
:€€€€€€€€€
V
ConstConst*
dtype0*
_output_shapes
:*
valueB"       
`
MeanMeanlogistic_lossConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
g

zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
w
logistic_loss_1/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
Ш
logistic_loss_1/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
™
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*'
_output_shapes
:€€€€€€€€€*
T0
j
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
•
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
v
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAdd
zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
y
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*
T0*'
_output_shapes
:€€€€€€€€€
f
logistic_loss_1/ExpExplogistic_loss_1/Select_1*
T0*'
_output_shapes
:€€€€€€€€€
e
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*
T0*'
_output_shapes
:€€€€€€€€€
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*'
_output_shapes
:€€€€€€€€€*
T0
X
Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
f
Mean_1Meanlogistic_loss_1Const_1*
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
discriminator_lossHistogramSummarydiscriminator_loss/tagadd*
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
ones_like_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
}
ones_like_1Fillones_like_1/Shapeones_like_1/Const*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
w
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
Ш
logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
™
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
j
logistic_loss_2/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
•
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
w
logistic_loss_2/mulMulDiscriminator/prob_1/BiasAddones_like_1*
T0*'
_output_shapes
:€€€€€€€€€
y
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*
T0*'
_output_shapes
:€€€€€€€€€
f
logistic_loss_2/ExpExplogistic_loss_2/Select_1*'
_output_shapes
:€€€€€€€€€*
T0
e
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*'
_output_shapes
:€€€€€€€€€*
T0
t
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*
T0*'
_output_shapes
:€€€€€€€€€
X
Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
f
Mean_2Meanlogistic_loss_2Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
a
generator_loss/tagConst*
valueB Bgenerator_loss*
dtype0*
_output_shapes
: 
_
generator_lossHistogramSummarygenerator_loss/tagMean_2*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  А?*
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
±
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Fill$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
≥
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
≠
gradients/Mean_grad/ReshapeReshape+gradients/add_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
f
gradients/Mean_grad/ShapeShapelogistic_loss*
T0*
out_type0*
_output_shapes
:
Ь
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
h
gradients/Mean_grad/Shape_1Shapelogistic_loss*
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
Ц
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ъ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
В
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
А
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
М
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
t
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
≥
gradients/Mean_1_grad/ReshapeReshape-gradients/add_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
j
gradients/Mean_1_grad/ShapeShapelogistic_loss_1*
_output_shapes
:*
T0*
out_type0
Ґ
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
l
gradients/Mean_1_grad/Shape_1Shapelogistic_loss_1*
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
Ь
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
g
gradients/Mean_1_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
†
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
gradients/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
И
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
Ж
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
В
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
Т
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*'
_output_shapes
:€€€€€€€€€*
T0
s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
out_type0*
_output_shapes
:*
T0
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:
“
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Є
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
µ
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Љ
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ї
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
Е
-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
В
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*'
_output_shapes
:€€€€€€€€€
И
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
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
Ў
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Њ
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ї
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
¬
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Л
/gradients/logistic_loss_1_grad/tuple/group_depsNoOp'^gradients/logistic_loss_1_grad/Reshape)^gradients/logistic_loss_1_grad/Reshape_1
К
7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Р
9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
z
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0*
_output_shapes
:
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
out_type0*
_output_shapes
:*
T0
ё
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Џ
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ѕ
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ё
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
≈
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
С
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
Т
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
Ш
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
І
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ь
&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*'
_output_shapes
:€€€€€€€€€*
T0
Х
-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*'
_output_shapes
:€€€€€€€€€*
T0
«
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€*
T0
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
д
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
а
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
«
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
д
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
Ћ
,gradients/logistic_loss_1/sub_grad/Reshape_1Reshape&gradients/logistic_loss_1/sub_grad/Neg*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Ч
3gradients/logistic_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/sub_grad/Reshape-^gradients/logistic_loss_1/sub_grad/Reshape_1
Ъ
;gradients/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/sub_grad/Reshape4^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/sub_grad/Reshape*'
_output_shapes
:€€€€€€€€€
†
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1
Ђ
*gradients/logistic_loss_1/Log1p_grad/add/xConst:^gradients/logistic_loss_1_grad/tuple/control_dependency_1*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ґ
(gradients/logistic_loss_1/Log1p_grad/addAdd*gradients/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*
T0*'
_output_shapes
:€€€€€€€€€
Щ
/gradients/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*
T0*'
_output_shapes
:€€€€€€€€€
Ќ
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€*
T0
Й
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
н
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*'
_output_shapes
:€€€€€€€€€*
T0
п
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€
Ш
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
Ь
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
Ґ
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:€€€€€€€€€
А
&gradients/logistic_loss/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
q
(gradients/logistic_loss/mul_grad/Shape_1Shape	ones_like*
T0*
out_type0*
_output_shapes
:
ё
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
•
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*
T0*'
_output_shapes
:€€€€€€€€€
…
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ѕ
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Є
&gradients/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€*
T0
ѕ
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
«
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
С
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
Т
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape
Ш
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
Ш
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0*'
_output_shapes
:€€€€€€€€€
Н
0gradients/logistic_loss_1/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
х
,gradients/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual;gradients/logistic_loss_1/sub_grad/tuple/control_dependency0gradients/logistic_loss_1/Select_grad/zeros_like*'
_output_shapes
:€€€€€€€€€*
T0
ч
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€*
T0
Ю
6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
§
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
™
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_grad/Select_1*'
_output_shapes
:€€€€€€€€€
Д
(gradients/logistic_loss_1/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
_output_shapes
:*
T0*
out_type0
t
*gradients/logistic_loss_1/mul_grad/Shape_1Shape
zeros_like*
T0*
out_type0*
_output_shapes
:
д
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
™
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1
zeros_like*'
_output_shapes
:€€€€€€€€€*
T0
ѕ
&gradients/logistic_loss_1/mul_grad/SumSum&gradients/logistic_loss_1/mul_grad/Mul8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
«
*gradients/logistic_loss_1/mul_grad/ReshapeReshape&gradients/logistic_loss_1/mul_grad/Sum(gradients/logistic_loss_1/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Њ
(gradients/logistic_loss_1/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€*
T0
’
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ќ
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Ч
3gradients/logistic_loss_1/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/mul_grad/Reshape-^gradients/logistic_loss_1/mul_grad/Reshape_1
Ъ
;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/mul_grad/Reshape4^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€
†
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
Ю
&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*'
_output_shapes
:€€€€€€€€€*
T0
В
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:€€€€€€€€€
№
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*'
_output_shapes
:€€€€€€€€€*
T0
ё
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:€€€€€€€€€
Ю
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
§
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select
™
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1*'
_output_shapes
:€€€€€€€€€
Ж
2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*'
_output_shapes
:€€€€€€€€€*
T0
д
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
ж
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*
T0*'
_output_shapes
:€€€€€€€€€
§
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
ђ
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select*'
_output_shapes
:€€€€€€€€€
≤
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1*'
_output_shapes
:€€€€€€€€€
Э
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€*
T0
°
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€*
T0
с
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:€€€€€€€€€
Р
5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:
Л
:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN6^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad
М
Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
≥
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
э
gradients/AddN_1AddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N*'
_output_shapes
:€€€€€€€€€
Ф
7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
T0*
data_formatNHWC*
_output_shapes
:
С
<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_18^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
Ф
Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
ї
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
ц
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
ц
1gradients/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	А*
transpose_a(*
transpose_b( 
І
9gradients/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp0^gradients/Discriminator/prob/MatMul_grad/MatMul2^gradients/Discriminator/prob/MatMul_grad/MatMul_1
±
Agradients/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity/gradients/Discriminator/prob/MatMul_grad/MatMul:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*B
_class8
64loc:@gradients/Discriminator/prob/MatMul_grad/MatMul
Ѓ
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	А
ъ
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(*
T0
ь
3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А*
transpose_a(*
transpose_b( *
T0
≠
;gradients/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp2^gradients/Discriminator/prob_1/MatMul_grad/MatMul4^gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
є
Cgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/Discriminator/prob_1/MatMul_grad/MatMul<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
ґ
Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	А
Ю
gradients/AddN_2AddNDgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
£
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
Ѓ
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
љ
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
Е
@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
З
:gradients/Discriminator/second_layer/leaky_relu_grad/zerosFill<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
г
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
Jgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Discriminator/second_layer/leaky_relu_grad/Shape<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Ї
;gradients/Discriminator/second_layer/leaky_relu_grad/SelectSelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency:gradients/Discriminator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
Љ
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
И
8gradients/Discriminator/second_layer/leaky_relu_grad/SumSum;gradients/Discriminator/second_layer/leaky_relu_grad/SelectJgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ю
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
О
:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1Lgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Д
>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
Ќ
Egradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp=^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape?^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
г
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
й
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
І
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
≤
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ѕ
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
З
Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosFill>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:€€€€€€€€€А*
T0
й
Cgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
†
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
¬
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
ƒ
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
О
:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Д
>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape:gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ф
<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
К
@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
”
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
л
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
с
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Э
gradients/AddN_3AddNCgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
N*
_output_shapes
:	А
Б
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
≤
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
¶
Ngradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
щ
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulMulMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
С
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ш
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ф
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Ч
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Pgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Р
Bgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ў
Igradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpA^gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeC^gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
б
Qgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeJ^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
щ
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А*
T0
Г
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
ґ
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
ђ
Pgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
€
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
Ч
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ю
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ъ
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Э
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Rgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ц
Dgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1Reshape@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
я
Kgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeE^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
й
Sgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeL^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape
Б
Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1L^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*W
_classM
KIloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А*
T0
Ќ
gradients/AddN_4AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А*
T0
Ђ
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:А
љ
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
”
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
Ф
\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
”
gradients/AddN_5AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
≠
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:А
Ѕ
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ў
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ь
^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А*
T0
Њ
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(*
T0
¶
Igradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_reluZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
п
Qgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpH^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulJ^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
С
Ygradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityGgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulR^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
П
[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
¬
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
ђ
Kgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
х
Sgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulL^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
Щ
[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulT^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul
Ч
]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
АА*
T0
з
gradients/AddN_6AddN\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:А
°
9gradients/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
ђ
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
‘
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2ShapeYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Д
?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Д
9gradients/Discriminator/first_layer/leaky_relu_grad/zerosFill;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:€€€€€€€€€А*
T0
а
@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Ч
Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ѕ
:gradients/Discriminator/first_layer/leaky_relu_grad/SelectSelect@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency9gradients/Discriminator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
—
<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Select@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqual9gradients/Discriminator/first_layer/leaky_relu_grad/zerosYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
Е
7gradients/Discriminator/first_layer/leaky_relu_grad/SumSum:gradients/Discriminator/first_layer/leaky_relu_grad/SelectIgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ы
;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape7gradients/Discriminator/first_layer/leaky_relu_grad/Sum9gradients/Discriminator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Л
9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Kgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Б
=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
 
Dgradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp<^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape>^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
я
Lgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeE^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape
е
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
•
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
∞
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ў
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
Ж
Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
К
;gradients/Discriminator/first_layer_1/leaky_relu_grad/zerosFill=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:€€€€€€€€€А*
T0
ж
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Э
Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
„
<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
ў
>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Л
9gradients/Discriminator/first_layer_1/leaky_relu_grad/SumSum<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectKgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Б
=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
С
;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
З
?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
–
Fgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp>^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape@^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
з
Ngradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeG^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
н
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
ж
gradients/AddN_7AddN[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
АА
А
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
∞
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
£
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ц
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
О
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
х
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
с
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Ф
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Ogradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
Agradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
÷
Hgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp@^gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeB^gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
Ё
Pgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeI^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape
х
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
В
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
і
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
©
Ogradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ь
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
Ф
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulOgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ы
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ч
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Qgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
У
Cgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1Reshape?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
№
Jgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeD^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
е
Rgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeK^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape
э
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
 
gradients/AddN_8AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
™
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:А
ї
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
–
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Р
[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradR^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А*
T0
–
gradients/AddN_9AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
ђ
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:А
њ
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
÷
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ш
]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
ї
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€Р*
transpose_a( *
transpose_b(
Х
Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/real_inYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
РА*
transpose_a(*
transpose_b( 
м
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
Н
Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€Р*
T0*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul
Л
Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
РА*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
њ
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:€€€€€€€€€Р*
transpose_a( *
transpose_b(*
T0
Э
Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
РА*
transpose_a(*
transpose_b( *
T0
т
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
Х
Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€Р*
T0
У
\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1S^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
РА
е
gradients/AddN_10AddN[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:А
д
gradients/AddN_11AddNZgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
РА
°
beta1_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
≤
beta1_power
VariableV2*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
—
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Н
beta1_power/readIdentitybeta1_power*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
°
beta2_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
≤
beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape: 
—
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
Н
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
н
WDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
„
MDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    
щ
GDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillWDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorMDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
РА
т
5Discriminator/first_layer/fully_connected/kernel/Adam
VariableV2*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
РА*
dtype0* 
_output_shapes
:
РА
я
<Discriminator/first_layer/fully_connected/kernel/Adam/AssignAssign5Discriminator/first_layer/fully_connected/kernel/AdamGDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
РА*
use_locking(
н
:Discriminator/first_layer/fully_connected/kernel/Adam/readIdentity5Discriminator/first_layer/fully_connected/kernel/Adam* 
_output_shapes
:
РА*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
п
YDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ў
ODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
€
IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillYDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
РА
ф
7Discriminator/first_layer/fully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
РА*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
РА
е
>Discriminator/first_layer/fully_connected/kernel/Adam_1/AssignAssign7Discriminator/first_layer/fully_connected/kernel/Adam_1IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
РА*
use_locking(
с
<Discriminator/first_layer/fully_connected/kernel/Adam_1/readIdentity7Discriminator/first_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
РА*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
„
EDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*
_output_shapes	
:А*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueBА*    *
dtype0
д
3Discriminator/first_layer/fully_connected/bias/Adam
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
“
:Discriminator/first_layer/fully_connected/bias/Adam/AssignAssign3Discriminator/first_layer/fully_connected/bias/AdamEDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
в
8Discriminator/first_layer/fully_connected/bias/Adam/readIdentity3Discriminator/first_layer/fully_connected/bias/Adam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:А
ў
GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ж
5Discriminator/first_layer/fully_connected/bias/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container 
Ў
<Discriminator/first_layer/fully_connected/bias/Adam_1/AssignAssign5Discriminator/first_layer/fully_connected/bias/Adam_1GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
ж
:Discriminator/first_layer/fully_connected/bias/Adam_1/readIdentity5Discriminator/first_layer/fully_connected/bias/Adam_1*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:А
п
XDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ў
NDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
э
HDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillXDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorNDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА
ф
6Discriminator/second_layer/fully_connected/kernel/Adam
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
г
=Discriminator/second_layer/fully_connected/kernel/Adam/AssignAssign6Discriminator/second_layer/fully_connected/kernel/AdamHDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
АА*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(
р
;Discriminator/second_layer/fully_connected/kernel/Adam/readIdentity6Discriminator/second_layer/fully_connected/kernel/Adam*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
с
ZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
џ
PDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorPDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА
ц
8Discriminator/second_layer/fully_connected/kernel/Adam_1
VariableV2*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
й
?Discriminator/second_layer/fully_connected/kernel/Adam_1/AssignAssign8Discriminator/second_layer/fully_connected/kernel/Adam_1JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
ф
=Discriminator/second_layer/fully_connected/kernel/Adam_1/readIdentity8Discriminator/second_layer/fully_connected/kernel/Adam_1*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
ў
FDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ж
4Discriminator/second_layer/fully_connected/bias/Adam
VariableV2*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
÷
;Discriminator/second_layer/fully_connected/bias/Adam/AssignAssign4Discriminator/second_layer/fully_connected/bias/AdamFDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
е
9Discriminator/second_layer/fully_connected/bias/Adam/readIdentity4Discriminator/second_layer/fully_connected/bias/Adam*
_output_shapes	
:А*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
џ
HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
и
6Discriminator/second_layer/fully_connected/bias/Adam_1
VariableV2*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
№
=Discriminator/second_layer/fully_connected/bias/Adam_1/AssignAssign6Discriminator/second_layer/fully_connected/bias/Adam_1HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
й
;Discriminator/second_layer/fully_connected/bias/Adam_1/readIdentity6Discriminator/second_layer/fully_connected/bias/Adam_1*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:А
µ
0Discriminator/prob/kernel/Adam/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
¬
Discriminator/prob/kernel/Adam
VariableV2*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	А*
dtype0*
_output_shapes
:	А
В
%Discriminator/prob/kernel/Adam/AssignAssignDiscriminator/prob/kernel/Adam0Discriminator/prob/kernel/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	А
І
#Discriminator/prob/kernel/Adam/readIdentityDiscriminator/prob/kernel/Adam*
_output_shapes
:	А*
T0*,
_class"
 loc:@Discriminator/prob/kernel
Ј
2Discriminator/prob/kernel/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	А*    *
dtype0*
_output_shapes
:	А
ƒ
 Discriminator/prob/kernel/Adam_1
VariableV2*
	container *
shape:	А*
dtype0*
_output_shapes
:	А*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel
И
'Discriminator/prob/kernel/Adam_1/AssignAssign Discriminator/prob/kernel/Adam_12Discriminator/prob/kernel/Adam_1/Initializer/zeros*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
Ђ
%Discriminator/prob/kernel/Adam_1/readIdentity Discriminator/prob/kernel/Adam_1*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	А
І
.Discriminator/prob/bias/Adam/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
і
Discriminator/prob/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container *
shape:
х
#Discriminator/prob/bias/Adam/AssignAssignDiscriminator/prob/bias/Adam.Discriminator/prob/bias/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
Ь
!Discriminator/prob/bias/Adam/readIdentityDiscriminator/prob/bias/Adam*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
©
0Discriminator/prob/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0
ґ
Discriminator/prob/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container *
shape:
ы
%Discriminator/prob/bias/Adam_1/AssignAssignDiscriminator/prob/bias/Adam_10Discriminator/prob/bias/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
†
#Discriminator/prob/bias/Adam_1/readIdentityDiscriminator/prob/bias/Adam_1*
_output_shapes
:*
T0**
_class 
loc:@Discriminator/prob/bias
W
Adam/learning_rateConst*
valueB
 *ЈQ9*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
э
FAdam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam0Discriminator/first_layer/fully_connected/kernel5Discriminator/first_layer/fully_connected/kernel/Adam7Discriminator/first_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_11*
use_locking( *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
РА
о
DAdam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam.Discriminator/first_layer/fully_connected/bias3Discriminator/first_layer/fully_connected/bias/Adam5Discriminator/first_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_10*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:А
Б
GAdam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernel6Discriminator/second_layer/fully_connected/kernel/Adam8Discriminator/second_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
АА
т
EAdam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/bias4Discriminator/second_layer/fully_connected/bias/Adam6Discriminator/second_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*
use_locking( *
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:А
И
/Adam/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernelDiscriminator/prob/kernel/Adam Discriminator/prob/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
use_nesterov( *
_output_shapes
:	А
щ
-Adam/update_Discriminator/prob/bias/ApplyAdam	ApplyAdamDiscriminator/prob/biasDiscriminator/prob/bias/AdamDiscriminator/prob/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
use_locking( *
T0**
_class 
loc:@Discriminator/prob/bias*
use_nesterov( *
_output_shapes
:
Х
Adam/mulMulbeta1_power/read
Adam/beta1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
є
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
Ч

Adam/mul_1Mulbeta2_power/read
Adam/beta2E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
љ
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
Ѓ
AdamNoOp^Adam/Assign^Adam/Assign_1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *
T0*

index_type0
v
%gradients_1/Mean_2_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ъ
gradients_1/Mean_2_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_2_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
l
gradients_1/Mean_2_grad/ShapeShapelogistic_loss_2*
out_type0*
_output_shapes
:*
T0
®
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€*

Tmultiples0
n
gradients_1/Mean_2_grad/Shape_1Shapelogistic_loss_2*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ґ
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
i
gradients_1/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
¶
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
c
!gradients_1/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
О
gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0
М
 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
Ж
gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
Ш
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*'
_output_shapes
:€€€€€€€€€*
T0
y
&gradients_1/logistic_loss_2_grad/ShapeShapelogistic_loss_2/sub*
_output_shapes
:*
T0*
out_type0
}
(gradients_1/logistic_loss_2_grad/Shape_1Shapelogistic_loss_2/Log1p*
T0*
out_type0*
_output_shapes
:
ё
6gradients_1/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_2_grad/Shape(gradients_1/logistic_loss_2_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ƒ
$gradients_1/logistic_loss_2_grad/SumSumgradients_1/Mean_2_grad/truediv6gradients_1/logistic_loss_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ѕ
(gradients_1/logistic_loss_2_grad/ReshapeReshape$gradients_1/logistic_loss_2_grad/Sum&gradients_1/logistic_loss_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
»
&gradients_1/logistic_loss_2_grad/Sum_1Sumgradients_1/Mean_2_grad/truediv8gradients_1/logistic_loss_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
«
*gradients_1/logistic_loss_2_grad/Reshape_1Reshape&gradients_1/logistic_loss_2_grad/Sum_1(gradients_1/logistic_loss_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
С
1gradients_1/logistic_loss_2_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_2_grad/Reshape+^gradients_1/logistic_loss_2_grad/Reshape_1
Т
9gradients_1/logistic_loss_2_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_2_grad/Reshape2^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_2_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Ш
;gradients_1/logistic_loss_2_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_2_grad/Reshape_12^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss_2_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
А
*gradients_1/logistic_loss_2/sub_grad/ShapeShapelogistic_loss_2/Select*
T0*
out_type0*
_output_shapes
:

,gradients_1/logistic_loss_2/sub_grad/Shape_1Shapelogistic_loss_2/mul*
out_type0*
_output_shapes
:*
T0
к
:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/sub_grad/Shape,gradients_1/logistic_loss_2/sub_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ж
(gradients_1/logistic_loss_2/sub_grad/SumSum9gradients_1/logistic_loss_2_grad/tuple/control_dependency:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ќ
,gradients_1/logistic_loss_2/sub_grad/ReshapeReshape(gradients_1/logistic_loss_2/sub_grad/Sum*gradients_1/logistic_loss_2/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
к
*gradients_1/logistic_loss_2/sub_grad/Sum_1Sum9gradients_1/logistic_loss_2_grad/tuple/control_dependency<gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
~
(gradients_1/logistic_loss_2/sub_grad/NegNeg*gradients_1/logistic_loss_2/sub_grad/Sum_1*
T0*
_output_shapes
:
—
.gradients_1/logistic_loss_2/sub_grad/Reshape_1Reshape(gradients_1/logistic_loss_2/sub_grad/Neg,gradients_1/logistic_loss_2/sub_grad/Shape_1*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
Э
5gradients_1/logistic_loss_2/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/sub_grad/Reshape/^gradients_1/logistic_loss_2/sub_grad/Reshape_1
Ґ
=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/sub_grad/Reshape6^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/sub_grad/Reshape*'
_output_shapes
:€€€€€€€€€
®
?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/sub_grad/Reshape_16^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/logistic_loss_2/sub_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€*
T0
ѓ
,gradients_1/logistic_loss_2/Log1p_grad/add/xConst<^gradients_1/logistic_loss_2_grad/tuple/control_dependency_1*
valueB
 *  А?*
dtype0*
_output_shapes
: 
¶
*gradients_1/logistic_loss_2/Log1p_grad/addAdd,gradients_1/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*'
_output_shapes
:€€€€€€€€€*
T0
Э
1gradients_1/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_2/Log1p_grad/add*'
_output_shapes
:€€€€€€€€€*
T0
”
*gradients_1/logistic_loss_2/Log1p_grad/mulMul;gradients_1/logistic_loss_2_grad/tuple/control_dependency_11gradients_1/logistic_loss_2/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€
П
2gradients_1/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
ы
.gradients_1/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_2/Select_grad/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
э
0gradients_1/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients_1/logistic_loss_2/Select_grad/zeros_like=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€
§
8gradients_1/logistic_loss_2/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_2/Select_grad/Select1^gradients_1/logistic_loss_2/Select_grad/Select_1
ђ
@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_2/Select_grad/Select9^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:€€€€€€€€€*
T0
≤
Bgradients_1/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_2/Select_grad/Select_19^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_grad/Select_1*'
_output_shapes
:€€€€€€€€€*
T0
Ж
*gradients_1/logistic_loss_2/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
_output_shapes
:*
T0*
out_type0
w
,gradients_1/logistic_loss_2/mul_grad/Shape_1Shapeones_like_1*
_output_shapes
:*
T0*
out_type0
к
:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/mul_grad/Shape,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ѓ
(gradients_1/logistic_loss_2/mul_grad/MulMul?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1ones_like_1*'
_output_shapes
:€€€€€€€€€*
T0
’
(gradients_1/logistic_loss_2/mul_grad/SumSum(gradients_1/logistic_loss_2/mul_grad/Mul:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ќ
,gradients_1/logistic_loss_2/mul_grad/ReshapeReshape(gradients_1/logistic_loss_2/mul_grad/Sum*gradients_1/logistic_loss_2/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
¬
*gradients_1/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€
џ
*gradients_1/logistic_loss_2/mul_grad/Sum_1Sum*gradients_1/logistic_loss_2/mul_grad/Mul_1<gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
”
.gradients_1/logistic_loss_2/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_2/mul_grad/Sum_1,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Э
5gradients_1/logistic_loss_2/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/mul_grad/Reshape/^gradients_1/logistic_loss_2/mul_grad/Reshape_1
Ґ
=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/mul_grad/Reshape6^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/mul_grad/Reshape
®
?gradients_1/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/mul_grad/Reshape_16^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/mul_grad/Reshape_1
Ґ
(gradients_1/logistic_loss_2/Exp_grad/mulMul*gradients_1/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*
T0*'
_output_shapes
:€€€€€€€€€
И
4gradients_1/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*'
_output_shapes
:€€€€€€€€€*
T0
к
0gradients_1/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual(gradients_1/logistic_loss_2/Exp_grad/mul4gradients_1/logistic_loss_2/Select_1_grad/zeros_like*
T0*'
_output_shapes
:€€€€€€€€€
м
2gradients_1/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual4gradients_1/logistic_loss_2/Select_1_grad/zeros_like(gradients_1/logistic_loss_2/Exp_grad/mul*
T0*'
_output_shapes
:€€€€€€€€€
™
:gradients_1/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_2/Select_1_grad/Select3^gradients_1/logistic_loss_2/Select_1_grad/Select_1
і
Bgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_2/Select_1_grad/Select;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_1_grad/Select
Ї
Dgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_2/Select_1_grad/Select_1;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/logistic_loss_2/Select_1_grad/Select_1*'
_output_shapes
:€€€€€€€€€
•
(gradients_1/logistic_loss_2/Neg_grad/NegNegBgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€*
T0
З
gradients_1/AddNAddN@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_2/Neg_grad/Neg*'
_output_shapes
:€€€€€€€€€*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*
N
Ц
9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes
:
Х
>gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN:^gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
Ъ
Fgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:€€€€€€€€€
√
Hgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ю
3gradients_1/Discriminator/prob_1/MatMul_grad/MatMulMatMulFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
А
5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А*
transpose_a(*
transpose_b( *
T0
≥
=gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul6^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1
Ѕ
Egradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Discriminator/prob_1/MatMul_grad/MatMul>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Њ
Ggradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	А
©
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
і
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
≈
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
Й
Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
У
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosFill@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:€€€€€€€€€А*
T0
л
Egradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
¶
Ngradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
 
?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
ћ
Agradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
Ф
<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SumSum?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectNgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
К
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ъ
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1SumAgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1Pgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Р
Bgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ў
Igradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOpA^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeC^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
у
Qgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeJ^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
щ
Sgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityBgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1J^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Е
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Є
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
≤
Rgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Г
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Э
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulRgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Д
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ю
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
£
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Tgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ь
Fgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
е
Mgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpE^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeG^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
с
Ugradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeN^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
Й
Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityFgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1N^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*Y
_classO
MKloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
џ
gradients_1/AddN_1AddNSgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N
±
Qgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
data_formatNHWC*
_output_shapes	
:А*
T0
«
Vgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1R^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
б
^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1W^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
§
`gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityQgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*d
_classZ
XVloc:@gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
∆
Kgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
∞
Mgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
ы
Ugradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpL^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulN^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
°
]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityKgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulV^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul
Я
_gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityMgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1V^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*`
_classV
TRloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
І
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
≤
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
№
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
И
Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Р
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zerosFill?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:€€€€€€€€€А*
T0
и
Dgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
£
Mgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
я
>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
б
@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
С
;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SumSum>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectMgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
Ч
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1Ogradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Н
Agradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
÷
Hgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp@^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeB^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
п
Pgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeI^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape
х
Rgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityAgradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1I^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Д
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
ґ
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
ѓ
Qgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
А
?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulQgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Б
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ы
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
†
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Sgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Щ
Egradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
в
Lgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpD^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeF^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
н
Tgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeM^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
Е
Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityEgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1M^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ў
gradients_1/AddN_2AddNRgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:€€€€€€€€€А*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
∞
Pgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:А
≈
Ugradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2Q^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ё
]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2V^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
†
_gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradV^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
√
Jgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€Р*
transpose_a( *
transpose_b(
°
Lgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
РА*
transpose_a(*
transpose_b( *
T0
ш
Tgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpK^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulM^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
Э
\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityJgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulU^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€Р
Ы
^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityLgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1U^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
РА
л
3gradients_1/Generator/fake_image/Tanh_grad/TanhGradTanhGradGenerator/fake_image/Tanh\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€Р
Ї
9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:Р
Є
>gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad4^gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
√
Fgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*(
_output_shapes
:€€€€€€€€€Р
ƒ
Hgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:Р
А
3gradients_1/Generator/fake_image/MatMul_grad/MatMulMatMulFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency Generator/fake_image/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
щ
5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1MatMulGenerator/last_layer/leaky_reluFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АР*
transpose_a(*
transpose_b( 
≥
=gradients_1/Generator/fake_image/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Generator/fake_image/MatMul_grad/MatMul6^gradients_1/Generator/fake_image/MatMul_grad/MatMul_1
Ѕ
Egradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/MatMul_grad/MatMul>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul
њ
Ggradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul_1* 
_output_shapes
:
АР
Щ
6gradients_1/Generator/last_layer/leaky_relu_grad/ShapeShape#Generator/last_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
∞
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
љ
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2ShapeEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
Б
<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ы
6gradients_1/Generator/last_layer/leaky_relu_grad/zerosFill8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
я
=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:€€€€€€€€€А*
T0
О
Fgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Generator/last_layer/leaky_relu_grad/Shape8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≤
7gradients_1/Generator/last_layer/leaky_relu_grad/SelectSelect=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency6gradients_1/Generator/last_layer/leaky_relu_grad/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
і
9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Select=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqual6gradients_1/Generator/last_layer/leaky_relu_grad/zerosEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
ь
4gradients_1/Generator/last_layer/leaky_relu_grad/SumSum7gradients_1/Generator/last_layer/leaky_relu_grad/SelectFgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
т
8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Generator/last_layer/leaky_relu_grad/Sum6gradients_1/Generator/last_layer/leaky_relu_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
В
6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Hgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ш
:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_18gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ѕ
Agradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape;^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1
”
Igradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeB^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
ў
Kgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1B^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
}
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
і
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
Ъ
Jgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ч
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulMulIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
Е
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/SumSum8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulJgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
м
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ж
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Mul%Generator/last_layer/leaky_relu/alphaIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Л
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Д
>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ќ
Egradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
—
Mgradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
й
Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
√
gradients_1/AddN_3AddNKgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*
N
«
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Generator/last_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Ь
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:А
ў
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Й
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_3_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
љ
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Н
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_3agradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ґ
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:А
М
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
Ј
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
∞
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:А*
T0
ї
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Generator/last_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ь
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:А*
dtype0*
_output_shapes
:
ў
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
£
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:€€€€€€€€€А
ƒ
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
љ
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
Ы
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Generator/last_layer/fully_connected/BiasAddbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
 
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ґ
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:А
М
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
Ј
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
∞
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:А
ё
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
Х
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg
ї
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:А
Ь
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:А
щ
Igradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:А*
T0
З
Ngradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad
∞
Vgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
Д
Xgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
Ц
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Generator/last_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
Ы
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Generator/last_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:А
Д
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Ґ
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:А*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul
®
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:А
∞
Cgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
Ъ
Egradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/third_layer/leaky_reluVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
г
Mgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1
Б
Ugradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
€
Wgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
э
gradients_1/AddN_4AddNdgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:А
Ѕ
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_43Generator/last_layer/batch_normalization/gamma/read*
_output_shapes	
:А*
T0
»
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_48Generator/last_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:А
ю
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Ъ
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:А*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul
†
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:А
Ы
7gradients_1/Generator/third_layer/leaky_relu_grad/ShapeShape$Generator/third_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
≤
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
ќ
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
В
=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ю
7gradients_1/Generator/third_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:€€€€€€€€€А*
T0
в
>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
С
Ggradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/third_layer/leaky_relu_grad/Shape9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≈
8gradients_1/Generator/third_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/third_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
«
:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/third_layer/leaky_relu_grad/zerosUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
€
5gradients_1/Generator/third_layer/leaky_relu_grad/SumSum8gradients_1/Generator/third_layer/leaky_relu_grad/SelectGgradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
х
9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/third_layer/leaky_relu_grad/Sum7gradients_1/Generator/third_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Е
7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Igradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ы
;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_19gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ƒ
Bgradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
„
Jgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
Ё
Lgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
~
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
ґ
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
Э
Kgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ъ
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
И
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
п
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
й
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/third_layer/leaky_relu/alphaJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
О
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
З
?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
–
Fgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
’
Ngradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*P
_classF
DBloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape
н
Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*R
_classH
FDloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
∆
gradients_1/AddN_5AddNLgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
…
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape9Generator/third_layer/batch_normalization/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
Э
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:А*
dtype0
№
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Л
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_5`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ј
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
П
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_5bgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
є
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:А*
T0*
Tshape0
П
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
ї
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
і
egradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:А
љ
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape-Generator/third_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Э
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:А*
dtype0*
_output_shapes
:
№
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
¶
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency7Generator/third_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:€€€€€€€€€А*
T0
«
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumSumNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ј
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Ю
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul-Generator/third_layer/fully_connected/BiasAddcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
Ќ
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1bgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
є
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:А
П
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
ї
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
і
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:А*
T0
а
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegNegegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
Ш
Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpf^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1M^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg
њ
agradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:А*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
†
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:А
ы
Jgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradcgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:А*
T0
К
Ogradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpd^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyK^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad
і
Wgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitycgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
И
Ygradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
Щ
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_17Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
Ю
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1:Generator/third_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:А
З
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulQ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
¶
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:А*
T0
ђ
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:А*
T0*c
_classY
WUloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
≥
Dgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
Э
Fgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1MatMul!Generator/second_layer/leaky_reluWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( *
T0
ж
Ngradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1
Е
Vgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Г
Xgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
АА*
T0*Y
_classO
MKloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1
А
gradients_1/AddN_6AddNegradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:А
√
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_64Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:А
 
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_69Generator/third_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:А
Б
Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpM^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Ю
agradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:А*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul
§
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:А
Э
8gradients_1/Generator/second_layer/leaky_relu_grad/ShapeShape%Generator/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
і
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
–
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2ShapeVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
Г
>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
8gradients_1/Generator/second_layer/leaky_relu_grad/zerosFill:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
е
?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:€€€€€€€€€А*
T0
Ф
Hgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/Generator/second_layer/leaky_relu_grad/Shape:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
…
9gradients_1/Generator/second_layer/leaky_relu_grad/SelectSelect?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency8gradients_1/Generator/second_layer/leaky_relu_grad/zeros*(
_output_shapes
:€€€€€€€€€А*
T0
Ћ
;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Select?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqual8gradients_1/Generator/second_layer/leaky_relu_grad/zerosVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
В
6gradients_1/Generator/second_layer/leaky_relu_grad/SumSum9gradients_1/Generator/second_layer/leaky_relu_grad/SelectHgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ш
:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeReshape6gradients_1/Generator/second_layer/leaky_relu_grad/Sum8gradients_1/Generator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
И
8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1Sum;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Jgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ю
<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1Reshape8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Cgradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_depsNoOp;^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape=^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1
џ
Kgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeD^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*M
_classC
A?loc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
б
Mgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1D^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Є
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
†
Lgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
э
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulMulKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:€€€€€€€€€А
Л
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/SumSum:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulLgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
т
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeReshape:gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
м
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Mul'Generator/second_layer/leaky_relu/alphaKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
С
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Ngradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
К
@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
”
Ggradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp?^gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeA^gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
ў
Ogradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeH^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
с
Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1H^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
…
gradients_1/AddN_7AddNMgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А
Ћ
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape:Generator/second_layer/batch_normalization/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
Ю
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:А*
dtype0*
_output_shapes
:
я
agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Н
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_7agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
√
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
С
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_7cgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Љ
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
Tshape0*
_output_shapes	
:А*
T0
Т
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
њ
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape
Є
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:А*
T0
њ
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape.Generator/second_layer/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ю
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:А*
dtype0*
_output_shapes
:
я
agradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
©
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:€€€€€€€€€А
 
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mulagradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
√
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
°
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul.Generator/second_layer/fully_connected/BiasAdddgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
–
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1cgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Љ
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:А*
T0*
Tshape0
Т
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
њ
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape
Є
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:А*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
в
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/NegNegfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
Ы
Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpg^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1N^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
√
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:А
§
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:А
э
Kgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGraddgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:А
Н
Pgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpe^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyL^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Є
Xgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitydgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
М
Zgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
Ь
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_18Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:А
°
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Muldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1;Generator/second_layer/batch_normalization/moving_mean/read*
_output_shapes	
:А*
T0
К
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulR^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
™
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:А*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul
∞
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:А*
T0*d
_classZ
XVloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
ґ
Egradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulMatMulXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency2Generator/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
Ю
Ggradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/first_layer/leaky_reluXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
АА*
transpose_a(*
transpose_b( 
й
Ogradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpF^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulH^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1
Й
Wgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityEgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulP^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
З
Ygradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityGgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1P^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
Г
gradients_1/AddN_8AddNfgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:А
≈
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_85Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:А
ћ
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_8:Generator/second_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:А*
T0
Д
Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpN^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Ґ
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:А
®
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:А*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Ы
7gradients_1/Generator/first_layer/leaky_relu_grad/ShapeShape$Generator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
¶
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
–
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2ShapeWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
В
=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
ю
7gradients_1/Generator/first_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
÷
>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
С
Ggradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/first_layer/leaky_relu_grad/Shape9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
«
8gradients_1/Generator/first_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:€€€€€€€€€А
…
:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/first_layer/leaky_relu_grad/zerosWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
€
5gradients_1/Generator/first_layer/leaky_relu_grad/SumSum8gradients_1/Generator/first_layer/leaky_relu_grad/SelectGgradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
х
9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/first_layer/leaky_relu_grad/Sum7gradients_1/Generator/first_layer/leaky_relu_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
Е
7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Igradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ы
;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_19gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ƒ
Bgradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
„
Jgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
Ё
Lgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
~
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
™
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Э
Kgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
о
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
И
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
п
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
й
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/first_layer/leaky_relu/alphaJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
О
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
З
?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
–
Fgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
’
Ngradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
н
Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*R
_classH
FDloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
∆
gradients_1/AddN_9AddNLgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:€€€€€€€€€А*
T0
™
Jgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:А
є
Ogradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_9K^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
ћ
Wgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_9P^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
И
Ygradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:А*
T0*]
_classS
QOloc:@gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
≤
Dgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/first_layer/fully_connected/kernel/read*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b(*
T0
Н
Fgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise_inWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	dА*
transpose_a(*
transpose_b( 
ж
Ngradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1
Д
Vgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€d
В
Xgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	dА
О
beta1_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Я
beta1_power_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@Generator/fake_image/bias
¬
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias
|
beta1_power_1/readIdentitybeta1_power_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
О
beta2_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
Я
beta2_power_1
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
¬
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
|
beta2_power_1/readIdentitybeta2_power_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
е
SGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   А   *
dtype0
ѕ
IGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
и
CGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	dА*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0
и
1Generator/first_layer/fully_connected/kernel/Adam
VariableV2*
	container *
shape:	dА*
dtype0*
_output_shapes
:	dА*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
ќ
8Generator/first_layer/fully_connected/kernel/Adam/AssignAssign1Generator/first_layer/fully_connected/kernel/AdamCGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	dА
а
6Generator/first_layer/fully_connected/kernel/Adam/readIdentity1Generator/first_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	dА
з
UGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   А   *
dtype0*
_output_shapes
:
—
KGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
о
EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	dА*
T0
к
3Generator/first_layer/fully_connected/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	dА*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	dА
‘
:Generator/first_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/first_layer/fully_connected/kernel/Adam_1EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	dА
д
8Generator/first_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/first_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	dА
ѕ
AGenerator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*
_output_shapes	
:А*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueBА*    *
dtype0
№
/Generator/first_layer/fully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:А
¬
6Generator/first_layer/fully_connected/bias/Adam/AssignAssign/Generator/first_layer/fully_connected/bias/AdamAGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
÷
4Generator/first_layer/fully_connected/bias/Adam/readIdentity/Generator/first_layer/fully_connected/bias/Adam*
_output_shapes	
:А*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
—
CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ё
1Generator/first_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:А
»
8Generator/first_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/first_layer/fully_connected/bias/Adam_1CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
Џ
6Generator/first_layer/fully_connected/bias/Adam_1/readIdentity1Generator/first_layer/fully_connected/bias/Adam_1*
_output_shapes	
:А*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
з
TGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"А      *
dtype0
—
JGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
н
DGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillTGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorJGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА*
T0
м
2Generator/second_layer/fully_connected/kernel/Adam
VariableV2*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
”
9Generator/second_layer/fully_connected/kernel/Adam/AssignAssign2Generator/second_layer/fully_connected/kernel/AdamDGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
д
7Generator/second_layer/fully_connected/kernel/Adam/readIdentity2Generator/second_layer/fully_connected/kernel/Adam*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА*
T0
й
VGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"А      *
dtype0*
_output_shapes
:
”
LGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
у
FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillVGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА
о
4Generator/second_layer/fully_connected/kernel/Adam_1
VariableV2*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
ў
;Generator/second_layer/fully_connected/kernel/Adam_1/AssignAssign4Generator/second_layer/fully_connected/kernel/Adam_1FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
АА*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
и
9Generator/second_layer/fully_connected/kernel/Adam_1/readIdentity4Generator/second_layer/fully_connected/kernel/Adam_1*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
АА
—
BGenerator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueBА*    
ё
0Generator/second_layer/fully_connected/bias/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container 
∆
7Generator/second_layer/fully_connected/bias/Adam/AssignAssign0Generator/second_layer/fully_connected/bias/AdamBGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
ў
5Generator/second_layer/fully_connected/bias/Adam/readIdentity0Generator/second_layer/fully_connected/bias/Adam*
_output_shapes	
:А*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias
”
DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
а
2Generator/second_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:А
ћ
9Generator/second_layer/fully_connected/bias/Adam_1/AssignAssign2Generator/second_layer/fully_connected/bias/Adam_1DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
Ё
7Generator/second_layer/fully_connected/bias/Adam_1/readIdentity2Generator/second_layer/fully_connected/bias/Adam_1*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:А
џ
GGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueBА*    *
dtype0*
_output_shapes	
:А
и
5Generator/second_layer/batch_normalization/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:А
Џ
<Generator/second_layer/batch_normalization/gamma/Adam/AssignAssign5Generator/second_layer/batch_normalization/gamma/AdamGGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
и
:Generator/second_layer/batch_normalization/gamma/Adam/readIdentity5Generator/second_layer/batch_normalization/gamma/Adam*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:А
Ё
IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueBА*    *
dtype0*
_output_shapes	
:А
к
7Generator/second_layer/batch_normalization/gamma/Adam_1
VariableV2*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:А*
dtype0*
_output_shapes	
:А
а
>Generator/second_layer/batch_normalization/gamma/Adam_1/AssignAssign7Generator/second_layer/batch_normalization/gamma/Adam_1IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А*
use_locking(
м
<Generator/second_layer/batch_normalization/gamma/Adam_1/readIdentity7Generator/second_layer/batch_normalization/gamma/Adam_1*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:А*
T0
ў
FGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
ж
4Generator/second_layer/batch_normalization/beta/Adam
VariableV2*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А
÷
;Generator/second_layer/batch_normalization/beta/Adam/AssignAssign4Generator/second_layer/batch_normalization/beta/AdamFGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
е
9Generator/second_layer/batch_normalization/beta/Adam/readIdentity4Generator/second_layer/batch_normalization/beta/Adam*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:А
џ
HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueBА*    
и
6Generator/second_layer/batch_normalization/beta/Adam_1
VariableV2*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А
№
=Generator/second_layer/batch_normalization/beta/Adam_1/AssignAssign6Generator/second_layer/batch_normalization/beta/Adam_1HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
й
;Generator/second_layer/batch_normalization/beta/Adam_1/readIdentity6Generator/second_layer/batch_normalization/beta/Adam_1*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:А
е
SGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ѕ
IGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    
й
CGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА
к
1Generator/third_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
АА
ѕ
8Generator/third_layer/fully_connected/kernel/Adam/AssignAssign1Generator/third_layer/fully_connected/kernel/AdamCGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
б
6Generator/third_layer/fully_connected/kernel/Adam/readIdentity1Generator/third_layer/fully_connected/kernel/Adam* 
_output_shapes
:
АА*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
з
UGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
—
KGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    
п
EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0
м
3Generator/third_layer/fully_connected/kernel/Adam_1
VariableV2*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА*
shared_name 
’
:Generator/third_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/third_layer/fully_connected/kernel/Adam_1EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
АА
е
8Generator/third_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/third_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
АА
ѕ
AGenerator/third_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
№
/Generator/third_layer/fully_connected/bias/Adam
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
¬
6Generator/third_layer/fully_connected/bias/Adam/AssignAssign/Generator/third_layer/fully_connected/bias/AdamAGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
÷
4Generator/third_layer/fully_connected/bias/Adam/readIdentity/Generator/third_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:А
—
CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
ё
1Generator/third_layer/fully_connected/bias/Adam_1
VariableV2*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
»
8Generator/third_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/third_layer/fully_connected/bias/Adam_1CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
Џ
6Generator/third_layer/fully_connected/bias/Adam_1/readIdentity1Generator/third_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:А
ў
FGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueBА*    *
dtype0*
_output_shapes	
:А
ж
4Generator/third_layer/batch_normalization/gamma/Adam
VariableV2*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:А*
dtype0*
_output_shapes	
:А
÷
;Generator/third_layer/batch_normalization/gamma/Adam/AssignAssign4Generator/third_layer/batch_normalization/gamma/AdamFGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
е
9Generator/third_layer/batch_normalization/gamma/Adam/readIdentity4Generator/third_layer/batch_normalization/gamma/Adam*
_output_shapes	
:А*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma
џ
HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueBА*    *
dtype0*
_output_shapes	
:А
и
6Generator/third_layer/batch_normalization/gamma/Adam_1
VariableV2*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:А*
dtype0*
_output_shapes	
:А
№
=Generator/third_layer/batch_normalization/gamma/Adam_1/AssignAssign6Generator/third_layer/batch_normalization/gamma/Adam_1HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
й
;Generator/third_layer/batch_normalization/gamma/Adam_1/readIdentity6Generator/third_layer/batch_normalization/gamma/Adam_1*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:А
„
EGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zerosConst*
_output_shapes	
:А*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueBА*    *
dtype0
д
3Generator/third_layer/batch_normalization/beta/Adam
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
“
:Generator/third_layer/batch_normalization/beta/Adam/AssignAssign3Generator/third_layer/batch_normalization/beta/AdamEGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
в
8Generator/third_layer/batch_normalization/beta/Adam/readIdentity3Generator/third_layer/batch_normalization/beta/Adam*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:А
ў
GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueBА*    *
dtype0*
_output_shapes	
:А
ж
5Generator/third_layer/batch_normalization/beta/Adam_1
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:А
Ў
<Generator/third_layer/batch_normalization/beta/Adam_1/AssignAssign5Generator/third_layer/batch_normalization/beta/Adam_1GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
_output_shapes	
:А*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(
ж
:Generator/third_layer/batch_normalization/beta/Adam_1/readIdentity5Generator/third_layer/batch_normalization/beta/Adam_1*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:А*
T0
г
RGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ќ
HGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
е
BGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zerosFillRGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
АА*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0
и
0Generator/last_layer/fully_connected/kernel/Adam
VariableV2*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
АА*
dtype0* 
_output_shapes
:
АА
Ћ
7Generator/last_layer/fully_connected/kernel/Adam/AssignAssign0Generator/last_layer/fully_connected/kernel/AdamBGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
АА*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(
ё
5Generator/last_layer/fully_connected/kernel/Adam/readIdentity0Generator/last_layer/fully_connected/kernel/Adam*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
АА
е
TGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ѕ
JGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0
л
DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillTGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorJGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
АА
к
2Generator/last_layer/fully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
АА*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
АА
—
9Generator/last_layer/fully_connected/kernel/Adam_1/AssignAssign2Generator/last_layer/fully_connected/kernel/Adam_1DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
АА*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(
в
7Generator/last_layer/fully_connected/kernel/Adam_1/readIdentity2Generator/last_layer/fully_connected/kernel/Adam_1*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
АА
ў
PGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:А*
dtype0
…
FGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Ў
@Generator/last_layer/fully_connected/bias/Adam/Initializer/zerosFillPGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorFGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/Const*
_output_shapes	
:А*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0
Џ
.Generator/last_layer/fully_connected/bias/Adam
VariableV2*
_output_shapes	
:А*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:А*
dtype0
Њ
5Generator/last_layer/fully_connected/bias/Adam/AssignAssign.Generator/last_layer/fully_connected/bias/Adam@Generator/last_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
”
3Generator/last_layer/fully_connected/bias/Adam/readIdentity.Generator/last_layer/fully_connected/bias/Adam*
_output_shapes	
:А*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
џ
RGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:А*
dtype0*
_output_shapes
:
Ћ
HGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
ё
BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zerosFillRGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/Const*
_output_shapes	
:А*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0
№
0Generator/last_layer/fully_connected/bias/Adam_1
VariableV2*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
ƒ
7Generator/last_layer/fully_connected/bias/Adam_1/AssignAssign0Generator/last_layer/fully_connected/bias/Adam_1BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:А
„
5Generator/last_layer/fully_connected/bias/Adam_1/readIdentity0Generator/last_layer/fully_connected/bias/Adam_1*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:А*
T0
г
UGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:А*
dtype0*
_output_shapes
:
”
KGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
м
EGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zerosFillUGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorKGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:А
д
3Generator/last_layer/batch_normalization/gamma/Adam
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container 
“
:Generator/last_layer/batch_normalization/gamma/Adam/AssignAssign3Generator/last_layer/batch_normalization/gamma/AdamEGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
в
8Generator/last_layer/batch_normalization/gamma/Adam/readIdentity3Generator/last_layer/batch_normalization/gamma/Adam*
_output_shapes	
:А*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma
е
WGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:А*
dtype0*
_output_shapes
:
’
MGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
т
GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zerosFillWGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorMGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/Const*
_output_shapes	
:А*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
ж
5Generator/last_layer/batch_normalization/gamma/Adam_1
VariableV2*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
Ў
<Generator/last_layer/batch_normalization/gamma/Adam_1/AssignAssign5Generator/last_layer/batch_normalization/gamma/Adam_1GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:А
ж
:Generator/last_layer/batch_normalization/gamma/Adam_1/readIdentity5Generator/last_layer/batch_normalization/gamma/Adam_1*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:А
б
TGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:А*
dtype0*
_output_shapes
:
—
JGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    
и
DGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zerosFillTGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorJGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:А
в
2Generator/last_layer/batch_normalization/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:А
ќ
9Generator/last_layer/batch_normalization/beta/Adam/AssignAssign2Generator/last_layer/batch_normalization/beta/AdamDGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
я
7Generator/last_layer/batch_normalization/beta/Adam/readIdentity2Generator/last_layer/batch_normalization/beta/Adam*
_output_shapes	
:А*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
г
VGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:А
”
LGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
о
FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zerosFillVGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:А
д
4Generator/last_layer/batch_normalization/beta/Adam_1
VariableV2*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:А*
dtype0*
_output_shapes	
:А
‘
;Generator/last_layer/batch_normalization/beta/Adam_1/AssignAssign4Generator/last_layer/batch_normalization/beta/Adam_1FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:А
г
9Generator/last_layer/batch_normalization/beta/Adam_1/readIdentity4Generator/last_layer/batch_normalization/beta/Adam_1*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:А
√
BGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
≠
8Generator/fake_image/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    
•
2Generator/fake_image/kernel/Adam/Initializer/zerosFillBGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensor8Generator/fake_image/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:
АР
»
 Generator/fake_image/kernel/Adam
VariableV2* 
_output_shapes
:
АР*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
АР*
dtype0
Л
'Generator/fake_image/kernel/Adam/AssignAssign Generator/fake_image/kernel/Adam2Generator/fake_image/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
АР*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel
Ѓ
%Generator/fake_image/kernel/Adam/readIdentity Generator/fake_image/kernel/Adam*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
АР
≈
DGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     
ѓ
:Generator/fake_image/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ђ
4Generator/fake_image/kernel/Adam_1/Initializer/zerosFillDGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensor:Generator/fake_image/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
АР*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0
 
"Generator/fake_image/kernel/Adam_1
VariableV2*
	container *
shape:
АР*
dtype0* 
_output_shapes
:
АР*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel
С
)Generator/fake_image/kernel/Adam_1/AssignAssign"Generator/fake_image/kernel/Adam_14Generator/fake_image/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
АР
≤
'Generator/fake_image/kernel/Adam_1/readIdentity"Generator/fake_image/kernel/Adam_1*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
АР
≠
0Generator/fake_image/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueBР*    *
dtype0*
_output_shapes	
:Р
Ї
Generator/fake_image/bias/Adam
VariableV2*
	container *
shape:Р*
dtype0*
_output_shapes	
:Р*
shared_name *,
_class"
 loc:@Generator/fake_image/bias
ю
%Generator/fake_image/bias/Adam/AssignAssignGenerator/fake_image/bias/Adam0Generator/fake_image/bias/Adam/Initializer/zeros*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:Р*
use_locking(*
T0
£
#Generator/fake_image/bias/Adam/readIdentityGenerator/fake_image/bias/Adam*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:Р*
T0
ѓ
2Generator/fake_image/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueBР*    *
dtype0*
_output_shapes	
:Р
Љ
 Generator/fake_image/bias/Adam_1
VariableV2*,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:Р*
dtype0*
_output_shapes	
:Р*
shared_name 
Д
'Generator/fake_image/bias/Adam_1/AssignAssign Generator/fake_image/bias/Adam_12Generator/fake_image/bias/Adam_1/Initializer/zeros*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:Р*
use_locking(*
T0
І
%Generator/fake_image/bias/Adam_1/readIdentity Generator/fake_image/bias/Adam_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:Р
Y
Adam_1/learning_rateConst*
valueB
 *ЈQ9*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
љ
DAdam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/first_layer/fully_connected/kernel1Generator/first_layer/fully_connected/kernel/Adam3Generator/first_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	dА
∞
BAdam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/first_layer/fully_connected/bias/Generator/first_layer/fully_connected/bias/Adam1Generator/first_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:А*
use_locking( 
ƒ
EAdam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam-Generator/second_layer/fully_connected/kernel2Generator/second_layer/fully_connected/kernel/Adam4Generator/second_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
АА
ґ
CAdam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam+Generator/second_layer/fully_connected/bias0Generator/second_layer/fully_connected/bias/Adam2Generator/second_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:А*
use_locking( 
ў
HAdam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam0Generator/second_layer/batch_normalization/gamma5Generator/second_layer/batch_normalization/gamma/Adam7Generator/second_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:А*
use_locking( *
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
use_nesterov( 
“
GAdam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam/Generator/second_layer/batch_normalization/beta4Generator/second_layer/batch_normalization/beta/Adam6Generator/second_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:А
Њ
DAdam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/third_layer/fully_connected/kernel1Generator/third_layer/fully_connected/kernel/Adam3Generator/third_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
АА*
use_locking( *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
use_nesterov( 
∞
BAdam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/third_layer/fully_connected/bias/Generator/third_layer/fully_connected/bias/Adam1Generator/third_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:А
”
GAdam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam/Generator/third_layer/batch_normalization/gamma4Generator/third_layer/batch_normalization/gamma/Adam6Generator/third_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:А
ћ
FAdam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdam	ApplyAdam.Generator/third_layer/batch_normalization/beta3Generator/third_layer/batch_normalization/beta/Adam5Generator/third_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0
Є
CAdam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Generator/last_layer/fully_connected/kernel0Generator/last_layer/fully_connected/kernel/Adam2Generator/last_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
АА*
use_locking( *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
™
AAdam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Generator/last_layer/fully_connected/bias.Generator/last_layer/fully_connected/bias/Adam0Generator/last_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:А*
use_locking( 
Ќ
FAdam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Generator/last_layer/batch_normalization/gamma3Generator/last_layer/batch_normalization/gamma/Adam5Generator/last_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:А
∆
EAdam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Generator/last_layer/batch_normalization/beta2Generator/last_layer/batch_normalization/beta/Adam4Generator/last_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:А*
use_locking( *
T0
Ў
3Adam_1/update_Generator/fake_image/kernel/ApplyAdam	ApplyAdamGenerator/fake_image/kernel Generator/fake_image/kernel/Adam"Generator/fake_image/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
АР*
use_locking( *
T0*.
_class$
" loc:@Generator/fake_image/kernel*
use_nesterov( 
 
1Adam_1/update_Generator/fake_image/bias/ApplyAdam	ApplyAdamGenerator/fake_image/biasGenerator/fake_image/bias/Adam Generator/fake_image/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
use_nesterov( *
_output_shapes	
:Р
’	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
™
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
„	
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta22^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
Ѓ
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
Г	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: ""
train_op

Adam
Adam_1"µЖ
	variables¶ЖҐЖ
з
.Generator/first_layer/fully_connected/kernel:03Generator/first_layer/fully_connected/kernel/Assign3Generator/first_layer/fully_connected/kernel/read:02IGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
÷
,Generator/first_layer/fully_connected/bias:01Generator/first_layer/fully_connected/bias/Assign1Generator/first_layer/fully_connected/bias/read:02>Generator/first_layer/fully_connected/bias/Initializer/zeros:08
л
/Generator/second_layer/fully_connected/kernel:04Generator/second_layer/fully_connected/kernel/Assign4Generator/second_layer/fully_connected/kernel/read:02JGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
Џ
-Generator/second_layer/fully_connected/bias:02Generator/second_layer/fully_connected/bias/Assign2Generator/second_layer/fully_connected/bias/read:02?Generator/second_layer/fully_connected/bias/Initializer/zeros:08
н
2Generator/second_layer/batch_normalization/gamma:07Generator/second_layer/batch_normalization/gamma/Assign7Generator/second_layer/batch_normalization/gamma/read:02CGenerator/second_layer/batch_normalization/gamma/Initializer/ones:08
к
1Generator/second_layer/batch_normalization/beta:06Generator/second_layer/batch_normalization/beta/Assign6Generator/second_layer/batch_normalization/beta/read:02CGenerator/second_layer/batch_normalization/beta/Initializer/zeros:08
Д
8Generator/second_layer/batch_normalization/moving_mean:0=Generator/second_layer/batch_normalization/moving_mean/Assign=Generator/second_layer/batch_normalization/moving_mean/read:02JGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros:0
У
<Generator/second_layer/batch_normalization/moving_variance:0AGenerator/second_layer/batch_normalization/moving_variance/AssignAGenerator/second_layer/batch_normalization/moving_variance/read:02MGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones:0
з
.Generator/third_layer/fully_connected/kernel:03Generator/third_layer/fully_connected/kernel/Assign3Generator/third_layer/fully_connected/kernel/read:02IGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform:08
÷
,Generator/third_layer/fully_connected/bias:01Generator/third_layer/fully_connected/bias/Assign1Generator/third_layer/fully_connected/bias/read:02>Generator/third_layer/fully_connected/bias/Initializer/zeros:08
й
1Generator/third_layer/batch_normalization/gamma:06Generator/third_layer/batch_normalization/gamma/Assign6Generator/third_layer/batch_normalization/gamma/read:02BGenerator/third_layer/batch_normalization/gamma/Initializer/ones:08
ж
0Generator/third_layer/batch_normalization/beta:05Generator/third_layer/batch_normalization/beta/Assign5Generator/third_layer/batch_normalization/beta/read:02BGenerator/third_layer/batch_normalization/beta/Initializer/zeros:08
А
7Generator/third_layer/batch_normalization/moving_mean:0<Generator/third_layer/batch_normalization/moving_mean/Assign<Generator/third_layer/batch_normalization/moving_mean/read:02IGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros:0
П
;Generator/third_layer/batch_normalization/moving_variance:0@Generator/third_layer/batch_normalization/moving_variance/Assign@Generator/third_layer/batch_normalization/moving_variance/read:02LGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones:0
г
-Generator/last_layer/fully_connected/kernel:02Generator/last_layer/fully_connected/kernel/Assign2Generator/last_layer/fully_connected/kernel/read:02HGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform:08
“
+Generator/last_layer/fully_connected/bias:00Generator/last_layer/fully_connected/bias/Assign0Generator/last_layer/fully_connected/bias/read:02=Generator/last_layer/fully_connected/bias/Initializer/zeros:08
е
0Generator/last_layer/batch_normalization/gamma:05Generator/last_layer/batch_normalization/gamma/Assign5Generator/last_layer/batch_normalization/gamma/read:02AGenerator/last_layer/batch_normalization/gamma/Initializer/ones:08
в
/Generator/last_layer/batch_normalization/beta:04Generator/last_layer/batch_normalization/beta/Assign4Generator/last_layer/batch_normalization/beta/read:02AGenerator/last_layer/batch_normalization/beta/Initializer/zeros:08
ь
6Generator/last_layer/batch_normalization/moving_mean:0;Generator/last_layer/batch_normalization/moving_mean/Assign;Generator/last_layer/batch_normalization/moving_mean/read:02HGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros:0
Л
:Generator/last_layer/batch_normalization/moving_variance:0?Generator/last_layer/batch_normalization/moving_variance/Assign?Generator/last_layer/batch_normalization/moving_variance/read:02KGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones:0
£
Generator/fake_image/kernel:0"Generator/fake_image/kernel/Assign"Generator/fake_image/kernel/read:028Generator/fake_image/kernel/Initializer/random_uniform:08
Т
Generator/fake_image/bias:0 Generator/fake_image/bias/Assign Generator/fake_image/bias/read:02-Generator/fake_image/bias/Initializer/zeros:08
ч
2Discriminator/first_layer/fully_connected/kernel:07Discriminator/first_layer/fully_connected/kernel/Assign7Discriminator/first_layer/fully_connected/kernel/read:02MDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
ж
0Discriminator/first_layer/fully_connected/bias:05Discriminator/first_layer/fully_connected/bias/Assign5Discriminator/first_layer/fully_connected/bias/read:02BDiscriminator/first_layer/fully_connected/bias/Initializer/zeros:08
ы
3Discriminator/second_layer/fully_connected/kernel:08Discriminator/second_layer/fully_connected/kernel/Assign8Discriminator/second_layer/fully_connected/kernel/read:02NDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
к
1Discriminator/second_layer/fully_connected/bias:06Discriminator/second_layer/fully_connected/bias/Assign6Discriminator/second_layer/fully_connected/bias/read:02CDiscriminator/second_layer/fully_connected/bias/Initializer/zeros:08
Ы
Discriminator/prob/kernel:0 Discriminator/prob/kernel/Assign Discriminator/prob/kernel/read:026Discriminator/prob/kernel/Initializer/random_uniform:08
К
Discriminator/prob/bias:0Discriminator/prob/bias/AssignDiscriminator/prob/bias/read:02+Discriminator/prob/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
А
7Discriminator/first_layer/fully_connected/kernel/Adam:0<Discriminator/first_layer/fully_connected/kernel/Adam/Assign<Discriminator/first_layer/fully_connected/kernel/Adam/read:02IDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros:0
И
9Discriminator/first_layer/fully_connected/kernel/Adam_1:0>Discriminator/first_layer/fully_connected/kernel/Adam_1/Assign>Discriminator/first_layer/fully_connected/kernel/Adam_1/read:02KDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ш
5Discriminator/first_layer/fully_connected/bias/Adam:0:Discriminator/first_layer/fully_connected/bias/Adam/Assign:Discriminator/first_layer/fully_connected/bias/Adam/read:02GDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros:0
А
7Discriminator/first_layer/fully_connected/bias/Adam_1:0<Discriminator/first_layer/fully_connected/bias/Adam_1/Assign<Discriminator/first_layer/fully_connected/bias/Adam_1/read:02IDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
Д
8Discriminator/second_layer/fully_connected/kernel/Adam:0=Discriminator/second_layer/fully_connected/kernel/Adam/Assign=Discriminator/second_layer/fully_connected/kernel/Adam/read:02JDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros:0
М
:Discriminator/second_layer/fully_connected/kernel/Adam_1:0?Discriminator/second_layer/fully_connected/kernel/Adam_1/Assign?Discriminator/second_layer/fully_connected/kernel/Adam_1/read:02LDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ь
6Discriminator/second_layer/fully_connected/bias/Adam:0;Discriminator/second_layer/fully_connected/bias/Adam/Assign;Discriminator/second_layer/fully_connected/bias/Adam/read:02HDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros:0
Д
8Discriminator/second_layer/fully_connected/bias/Adam_1:0=Discriminator/second_layer/fully_connected/bias/Adam_1/Assign=Discriminator/second_layer/fully_connected/bias/Adam_1/read:02JDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
§
 Discriminator/prob/kernel/Adam:0%Discriminator/prob/kernel/Adam/Assign%Discriminator/prob/kernel/Adam/read:022Discriminator/prob/kernel/Adam/Initializer/zeros:0
ђ
"Discriminator/prob/kernel/Adam_1:0'Discriminator/prob/kernel/Adam_1/Assign'Discriminator/prob/kernel/Adam_1/read:024Discriminator/prob/kernel/Adam_1/Initializer/zeros:0
Ь
Discriminator/prob/bias/Adam:0#Discriminator/prob/bias/Adam/Assign#Discriminator/prob/bias/Adam/read:020Discriminator/prob/bias/Adam/Initializer/zeros:0
§
 Discriminator/prob/bias/Adam_1:0%Discriminator/prob/bias/Adam_1/Assign%Discriminator/prob/bias/Adam_1/read:022Discriminator/prob/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
р
3Generator/first_layer/fully_connected/kernel/Adam:08Generator/first_layer/fully_connected/kernel/Adam/Assign8Generator/first_layer/fully_connected/kernel/Adam/read:02EGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ш
5Generator/first_layer/fully_connected/kernel/Adam_1:0:Generator/first_layer/fully_connected/kernel/Adam_1/Assign:Generator/first_layer/fully_connected/kernel/Adam_1/read:02GGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
и
1Generator/first_layer/fully_connected/bias/Adam:06Generator/first_layer/fully_connected/bias/Adam/Assign6Generator/first_layer/fully_connected/bias/Adam/read:02CGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros:0
р
3Generator/first_layer/fully_connected/bias/Adam_1:08Generator/first_layer/fully_connected/bias/Adam_1/Assign8Generator/first_layer/fully_connected/bias/Adam_1/read:02EGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
ф
4Generator/second_layer/fully_connected/kernel/Adam:09Generator/second_layer/fully_connected/kernel/Adam/Assign9Generator/second_layer/fully_connected/kernel/Adam/read:02FGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ь
6Generator/second_layer/fully_connected/kernel/Adam_1:0;Generator/second_layer/fully_connected/kernel/Adam_1/Assign;Generator/second_layer/fully_connected/kernel/Adam_1/read:02HGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
м
2Generator/second_layer/fully_connected/bias/Adam:07Generator/second_layer/fully_connected/bias/Adam/Assign7Generator/second_layer/fully_connected/bias/Adam/read:02DGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros:0
ф
4Generator/second_layer/fully_connected/bias/Adam_1:09Generator/second_layer/fully_connected/bias/Adam_1/Assign9Generator/second_layer/fully_connected/bias/Adam_1/read:02FGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
А
7Generator/second_layer/batch_normalization/gamma/Adam:0<Generator/second_layer/batch_normalization/gamma/Adam/Assign<Generator/second_layer/batch_normalization/gamma/Adam/read:02IGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros:0
И
9Generator/second_layer/batch_normalization/gamma/Adam_1:0>Generator/second_layer/batch_normalization/gamma/Adam_1/Assign>Generator/second_layer/batch_normalization/gamma/Adam_1/read:02KGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
ь
6Generator/second_layer/batch_normalization/beta/Adam:0;Generator/second_layer/batch_normalization/beta/Adam/Assign;Generator/second_layer/batch_normalization/beta/Adam/read:02HGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros:0
Д
8Generator/second_layer/batch_normalization/beta/Adam_1:0=Generator/second_layer/batch_normalization/beta/Adam_1/Assign=Generator/second_layer/batch_normalization/beta/Adam_1/read:02JGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
р
3Generator/third_layer/fully_connected/kernel/Adam:08Generator/third_layer/fully_connected/kernel/Adam/Assign8Generator/third_layer/fully_connected/kernel/Adam/read:02EGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ш
5Generator/third_layer/fully_connected/kernel/Adam_1:0:Generator/third_layer/fully_connected/kernel/Adam_1/Assign:Generator/third_layer/fully_connected/kernel/Adam_1/read:02GGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
и
1Generator/third_layer/fully_connected/bias/Adam:06Generator/third_layer/fully_connected/bias/Adam/Assign6Generator/third_layer/fully_connected/bias/Adam/read:02CGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros:0
р
3Generator/third_layer/fully_connected/bias/Adam_1:08Generator/third_layer/fully_connected/bias/Adam_1/Assign8Generator/third_layer/fully_connected/bias/Adam_1/read:02EGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
ь
6Generator/third_layer/batch_normalization/gamma/Adam:0;Generator/third_layer/batch_normalization/gamma/Adam/Assign;Generator/third_layer/batch_normalization/gamma/Adam/read:02HGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros:0
Д
8Generator/third_layer/batch_normalization/gamma/Adam_1:0=Generator/third_layer/batch_normalization/gamma/Adam_1/Assign=Generator/third_layer/batch_normalization/gamma/Adam_1/read:02JGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
ш
5Generator/third_layer/batch_normalization/beta/Adam:0:Generator/third_layer/batch_normalization/beta/Adam/Assign:Generator/third_layer/batch_normalization/beta/Adam/read:02GGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros:0
А
7Generator/third_layer/batch_normalization/beta/Adam_1:0<Generator/third_layer/batch_normalization/beta/Adam_1/Assign<Generator/third_layer/batch_normalization/beta/Adam_1/read:02IGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
м
2Generator/last_layer/fully_connected/kernel/Adam:07Generator/last_layer/fully_connected/kernel/Adam/Assign7Generator/last_layer/fully_connected/kernel/Adam/read:02DGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ф
4Generator/last_layer/fully_connected/kernel/Adam_1:09Generator/last_layer/fully_connected/kernel/Adam_1/Assign9Generator/last_layer/fully_connected/kernel/Adam_1/read:02FGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
д
0Generator/last_layer/fully_connected/bias/Adam:05Generator/last_layer/fully_connected/bias/Adam/Assign5Generator/last_layer/fully_connected/bias/Adam/read:02BGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros:0
м
2Generator/last_layer/fully_connected/bias/Adam_1:07Generator/last_layer/fully_connected/bias/Adam_1/Assign7Generator/last_layer/fully_connected/bias/Adam_1/read:02DGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
ш
5Generator/last_layer/batch_normalization/gamma/Adam:0:Generator/last_layer/batch_normalization/gamma/Adam/Assign:Generator/last_layer/batch_normalization/gamma/Adam/read:02GGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros:0
А
7Generator/last_layer/batch_normalization/gamma/Adam_1:0<Generator/last_layer/batch_normalization/gamma/Adam_1/Assign<Generator/last_layer/batch_normalization/gamma/Adam_1/read:02IGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
ф
4Generator/last_layer/batch_normalization/beta/Adam:09Generator/last_layer/batch_normalization/beta/Adam/Assign9Generator/last_layer/batch_normalization/beta/Adam/read:02FGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros:0
ь
6Generator/last_layer/batch_normalization/beta/Adam_1:0;Generator/last_layer/batch_normalization/beta/Adam_1/Assign;Generator/last_layer/batch_normalization/beta/Adam_1/read:02HGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
ђ
"Generator/fake_image/kernel/Adam:0'Generator/fake_image/kernel/Adam/Assign'Generator/fake_image/kernel/Adam/read:024Generator/fake_image/kernel/Adam/Initializer/zeros:0
і
$Generator/fake_image/kernel/Adam_1:0)Generator/fake_image/kernel/Adam_1/Assign)Generator/fake_image/kernel/Adam_1/read:026Generator/fake_image/kernel/Adam_1/Initializer/zeros:0
§
 Generator/fake_image/bias/Adam:0%Generator/fake_image/bias/Adam/Assign%Generator/fake_image/bias/Adam/read:022Generator/fake_image/bias/Adam/Initializer/zeros:0
ђ
"Generator/fake_image/bias/Adam_1:0'Generator/fake_image/bias/Adam_1/Assign'Generator/fake_image/bias/Adam_1/read:024Generator/fake_image/bias/Adam_1/Initializer/zeros:0"7
	summaries*
(
discriminator_loss:0
generator_loss:0"Џ%
trainable_variables¬%њ%
з
.Generator/first_layer/fully_connected/kernel:03Generator/first_layer/fully_connected/kernel/Assign3Generator/first_layer/fully_connected/kernel/read:02IGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
÷
,Generator/first_layer/fully_connected/bias:01Generator/first_layer/fully_connected/bias/Assign1Generator/first_layer/fully_connected/bias/read:02>Generator/first_layer/fully_connected/bias/Initializer/zeros:08
л
/Generator/second_layer/fully_connected/kernel:04Generator/second_layer/fully_connected/kernel/Assign4Generator/second_layer/fully_connected/kernel/read:02JGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
Џ
-Generator/second_layer/fully_connected/bias:02Generator/second_layer/fully_connected/bias/Assign2Generator/second_layer/fully_connected/bias/read:02?Generator/second_layer/fully_connected/bias/Initializer/zeros:08
н
2Generator/second_layer/batch_normalization/gamma:07Generator/second_layer/batch_normalization/gamma/Assign7Generator/second_layer/batch_normalization/gamma/read:02CGenerator/second_layer/batch_normalization/gamma/Initializer/ones:08
к
1Generator/second_layer/batch_normalization/beta:06Generator/second_layer/batch_normalization/beta/Assign6Generator/second_layer/batch_normalization/beta/read:02CGenerator/second_layer/batch_normalization/beta/Initializer/zeros:08
з
.Generator/third_layer/fully_connected/kernel:03Generator/third_layer/fully_connected/kernel/Assign3Generator/third_layer/fully_connected/kernel/read:02IGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform:08
÷
,Generator/third_layer/fully_connected/bias:01Generator/third_layer/fully_connected/bias/Assign1Generator/third_layer/fully_connected/bias/read:02>Generator/third_layer/fully_connected/bias/Initializer/zeros:08
й
1Generator/third_layer/batch_normalization/gamma:06Generator/third_layer/batch_normalization/gamma/Assign6Generator/third_layer/batch_normalization/gamma/read:02BGenerator/third_layer/batch_normalization/gamma/Initializer/ones:08
ж
0Generator/third_layer/batch_normalization/beta:05Generator/third_layer/batch_normalization/beta/Assign5Generator/third_layer/batch_normalization/beta/read:02BGenerator/third_layer/batch_normalization/beta/Initializer/zeros:08
г
-Generator/last_layer/fully_connected/kernel:02Generator/last_layer/fully_connected/kernel/Assign2Generator/last_layer/fully_connected/kernel/read:02HGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform:08
“
+Generator/last_layer/fully_connected/bias:00Generator/last_layer/fully_connected/bias/Assign0Generator/last_layer/fully_connected/bias/read:02=Generator/last_layer/fully_connected/bias/Initializer/zeros:08
е
0Generator/last_layer/batch_normalization/gamma:05Generator/last_layer/batch_normalization/gamma/Assign5Generator/last_layer/batch_normalization/gamma/read:02AGenerator/last_layer/batch_normalization/gamma/Initializer/ones:08
в
/Generator/last_layer/batch_normalization/beta:04Generator/last_layer/batch_normalization/beta/Assign4Generator/last_layer/batch_normalization/beta/read:02AGenerator/last_layer/batch_normalization/beta/Initializer/zeros:08
£
Generator/fake_image/kernel:0"Generator/fake_image/kernel/Assign"Generator/fake_image/kernel/read:028Generator/fake_image/kernel/Initializer/random_uniform:08
Т
Generator/fake_image/bias:0 Generator/fake_image/bias/Assign Generator/fake_image/bias/read:02-Generator/fake_image/bias/Initializer/zeros:08
ч
2Discriminator/first_layer/fully_connected/kernel:07Discriminator/first_layer/fully_connected/kernel/Assign7Discriminator/first_layer/fully_connected/kernel/read:02MDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
ж
0Discriminator/first_layer/fully_connected/bias:05Discriminator/first_layer/fully_connected/bias/Assign5Discriminator/first_layer/fully_connected/bias/read:02BDiscriminator/first_layer/fully_connected/bias/Initializer/zeros:08
ы
3Discriminator/second_layer/fully_connected/kernel:08Discriminator/second_layer/fully_connected/kernel/Assign8Discriminator/second_layer/fully_connected/kernel/read:02NDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
к
1Discriminator/second_layer/fully_connected/bias:06Discriminator/second_layer/fully_connected/bias/Assign6Discriminator/second_layer/fully_connected/bias/read:02CDiscriminator/second_layer/fully_connected/bias/Initializer/zeros:08
Ы
Discriminator/prob/kernel:0 Discriminator/prob/kernel/Assign Discriminator/prob/kernel/read:026Discriminator/prob/kernel/Initializer/random_uniform:08
К
Discriminator/prob/bias:0Discriminator/prob/bias/AssignDiscriminator/prob/bias/read:02+Discriminator/prob/bias/Initializer/zeros:08£ид