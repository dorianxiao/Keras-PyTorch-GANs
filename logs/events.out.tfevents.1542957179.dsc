       ЃK"	  Ръ§жAbrain.Event:2p`ъ	     фgџ	я§ъ§жA"і
u
Generator/noise_inPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџd*
shape:џџџџџџџџџd
п
MGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
б
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&О*
dtype0*
_output_shapes
: 
б
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&>*
dtype0*
_output_shapes
: 
Ц
UGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*

seed *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
seed2 *
dtype0*
_output_shapes
:	d
Ю
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
: 
с
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
г
GGenerator/first_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
у
,Generator/first_layer/fully_connected/kernel
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d
Ш
3Generator/first_layer/fully_connected/kernel/AssignAssign,Generator/first_layer/fully_connected/kernelGGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d
ж
1Generator/first_layer/fully_connected/kernel/readIdentity,Generator/first_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
Ъ
<Generator/first_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    
з
*Generator/first_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:
Г
1Generator/first_layer/fully_connected/bias/AssignAssign*Generator/first_layer/fully_connected/bias<Generator/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ь
/Generator/first_layer/fully_connected/bias/readIdentity*Generator/first_layer/fully_connected/bias*
_output_shapes	
:*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
ж
,Generator/first_layer/fully_connected/MatMulMatMulGenerator/noise_in1Generator/first_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
с
-Generator/first_layer/fully_connected/BiasAddBiasAdd,Generator/first_layer/fully_connected/MatMul/Generator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
k
&Generator/first_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>
Е
$Generator/first_layer/leaky_relu/mulMul&Generator/first_layer/leaky_relu/alpha-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Г
 Generator/first_layer/leaky_reluMaximum$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
с
NGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
г
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   О*
dtype0*
_output_shapes
: 
г
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
Ъ
VGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformNGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*

seed *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:

в
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
_output_shapes
: 
ц
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulVGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

и
HGenerator/second_layer/fully_connected/kernel/Initializer/random_uniformAddLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

ч
-Generator/second_layer/fully_connected/kernel
VariableV2*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Э
4Generator/second_layer/fully_connected/kernel/AssignAssign-Generator/second_layer/fully_connected/kernelHGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

к
2Generator/second_layer/fully_connected/kernel/readIdentity-Generator/second_layer/fully_connected/kernel*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

Ь
=Generator/second_layer/fully_connected/bias/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
й
+Generator/second_layer/fully_connected/bias
VariableV2*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
З
2Generator/second_layer/fully_connected/bias/AssignAssign+Generator/second_layer/fully_connected/bias=Generator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Я
0Generator/second_layer/fully_connected/bias/readIdentity+Generator/second_layer/fully_connected/bias*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:
ц
-Generator/second_layer/fully_connected/MatMulMatMul Generator/first_layer/leaky_relu2Generator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
ф
.Generator/second_layer/fully_connected/BiasAddBiasAdd-Generator/second_layer/fully_connected/MatMul0Generator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
е
AGenerator/second_layer/batch_normalization/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*  ?
у
0Generator/second_layer/batch_normalization/gamma
VariableV2*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ъ
7Generator/second_layer/batch_normalization/gamma/AssignAssign0Generator/second_layer/batch_normalization/gammaAGenerator/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
о
5Generator/second_layer/batch_normalization/gamma/readIdentity0Generator/second_layer/batch_normalization/gamma*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:
д
AGenerator/second_layer/batch_normalization/beta/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
с
/Generator/second_layer/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:
Ч
6Generator/second_layer/batch_normalization/beta/AssignAssign/Generator/second_layer/batch_normalization/betaAGenerator/second_layer/batch_normalization/beta/Initializer/zeros*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
л
4Generator/second_layer/batch_normalization/beta/readIdentity/Generator/second_layer/batch_normalization/beta*
_output_shapes	
:*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
т
HGenerator/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
я
6Generator/second_layer/batch_normalization/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
	container 
у
=Generator/second_layer/batch_normalization/moving_mean/AssignAssign6Generator/second_layer/batch_normalization/moving_meanHGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:
№
;Generator/second_layer/batch_normalization/moving_mean/readIdentity6Generator/second_layer/batch_normalization/moving_mean*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
_output_shapes	
:
щ
KGenerator/second_layer/batch_normalization/moving_variance/Initializer/onesConst*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ї
:Generator/second_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
	container *
shape:
ђ
AGenerator/second_layer/batch_normalization/moving_variance/AssignAssign:Generator/second_layer/batch_normalization/moving_varianceKGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:
ќ
?Generator/second_layer/batch_normalization/moving_variance/readIdentity:Generator/second_layer/batch_normalization/moving_variance*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
_output_shapes	
:

:Generator/second_layer/batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:
т
8Generator/second_layer/batch_normalization/batchnorm/addAdd?Generator/second_layer/batch_normalization/moving_variance/read:Generator/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:
Ѓ
:Generator/second_layer/batch_normalization/batchnorm/RsqrtRsqrt8Generator/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
и
8Generator/second_layer/batch_normalization/batchnorm/mulMul:Generator/second_layer/batch_normalization/batchnorm/Rsqrt5Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
о
:Generator/second_layer/batch_normalization/batchnorm/mul_1Mul.Generator/second_layer/fully_connected/BiasAdd8Generator/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:џџџџџџџџџ*
T0
о
:Generator/second_layer/batch_normalization/batchnorm/mul_2Mul;Generator/second_layer/batch_normalization/moving_mean/read8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
з
8Generator/second_layer/batch_normalization/batchnorm/subSub4Generator/second_layer/batch_normalization/beta/read:Generator/second_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:*
T0
ъ
:Generator/second_layer/batch_normalization/batchnorm/add_1Add:Generator/second_layer/batch_normalization/batchnorm/mul_18Generator/second_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:џџџџџџџџџ
l
'Generator/second_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ф
%Generator/second_layer/leaky_relu/mulMul'Generator/second_layer/leaky_relu/alpha:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
Т
!Generator/second_layer/leaky_reluMaximum%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
п
MGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
б
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *ѓЕН
б
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *ѓЕ=
Ч
UGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
seed2 
Ю
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
_output_shapes
: 
т
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

д
GGenerator/third_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
х
,Generator/third_layer/fully_connected/kernel
VariableV2*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Щ
3Generator/third_layer/fully_connected/kernel/AssignAssign,Generator/third_layer/fully_connected/kernelGGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

з
1Generator/third_layer/fully_connected/kernel/readIdentity,Generator/third_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

Ъ
<Generator/third_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    
з
*Generator/third_layer/fully_connected/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container 
Г
1Generator/third_layer/fully_connected/bias/AssignAssign*Generator/third_layer/fully_connected/bias<Generator/third_layer/fully_connected/bias/Initializer/zeros*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ь
/Generator/third_layer/fully_connected/bias/readIdentity*Generator/third_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:
х
,Generator/third_layer/fully_connected/MatMulMatMul!Generator/second_layer/leaky_relu1Generator/third_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
с
-Generator/third_layer/fully_connected/BiasAddBiasAdd,Generator/third_layer/fully_connected/MatMul/Generator/third_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
г
@Generator/third_layer/batch_normalization/gamma/Initializer/onesConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
с
/Generator/third_layer/batch_normalization/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container 
Ц
6Generator/third_layer/batch_normalization/gamma/AssignAssign/Generator/third_layer/batch_normalization/gamma@Generator/third_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
л
4Generator/third_layer/batch_normalization/gamma/readIdentity/Generator/third_layer/batch_normalization/gamma*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
в
@Generator/third_layer/batch_normalization/beta/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
п
.Generator/third_layer/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:
У
5Generator/third_layer/batch_normalization/beta/AssignAssign.Generator/third_layer/batch_normalization/beta@Generator/third_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
и
3Generator/third_layer/batch_normalization/beta/readIdentity.Generator/third_layer/batch_normalization/beta*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:
р
GGenerator/third_layer/batch_normalization/moving_mean/Initializer/zerosConst*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
э
5Generator/third_layer/batch_normalization/moving_mean
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean
п
<Generator/third_layer/batch_normalization/moving_mean/AssignAssign5Generator/third_layer/batch_normalization/moving_meanGGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:
э
:Generator/third_layer/batch_normalization/moving_mean/readIdentity5Generator/third_layer/batch_normalization/moving_mean*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
_output_shapes	
:*
T0
ч
JGenerator/third_layer/batch_normalization/moving_variance/Initializer/onesConst*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ѕ
9Generator/third_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
	container *
shape:
ю
@Generator/third_layer/batch_normalization/moving_variance/AssignAssign9Generator/third_layer/batch_normalization/moving_varianceJGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
љ
>Generator/third_layer/batch_normalization/moving_variance/readIdentity9Generator/third_layer/batch_normalization/moving_variance*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
_output_shapes	
:
~
9Generator/third_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
п
7Generator/third_layer/batch_normalization/batchnorm/addAdd>Generator/third_layer/batch_normalization/moving_variance/read9Generator/third_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:
Ё
9Generator/third_layer/batch_normalization/batchnorm/RsqrtRsqrt7Generator/third_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
е
7Generator/third_layer/batch_normalization/batchnorm/mulMul9Generator/third_layer/batch_normalization/batchnorm/Rsqrt4Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
л
9Generator/third_layer/batch_normalization/batchnorm/mul_1Mul-Generator/third_layer/fully_connected/BiasAdd7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
л
9Generator/third_layer/batch_normalization/batchnorm/mul_2Mul:Generator/third_layer/batch_normalization/moving_mean/read7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
д
7Generator/third_layer/batch_normalization/batchnorm/subSub3Generator/third_layer/batch_normalization/beta/read9Generator/third_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ч
9Generator/third_layer/batch_normalization/batchnorm/add_1Add9Generator/third_layer/batch_normalization/batchnorm/mul_17Generator/third_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:џџџџџџџџџ*
T0
k
&Generator/third_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>
С
$Generator/third_layer/leaky_relu/mulMul&Generator/third_layer/leaky_relu/alpha9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
П
 Generator/third_layer/leaky_reluMaximum$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
н
LGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Я
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  Н*
dtype0*
_output_shapes
: 
Я
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
Ф
TGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shape*

seed *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:

Ъ
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/subSubJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
_output_shapes
: 
о
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
а
FGenerator/last_layer/fully_connected/kernel/Initializer/random_uniformAddJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

у
+Generator/last_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:

Х
2Generator/last_layer/fully_connected/kernel/AssignAssign+Generator/last_layer/fully_connected/kernelFGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
д
0Generator/last_layer/fully_connected/kernel/readIdentity+Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
д
KGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:
Ф
AGenerator/last_layer/fully_connected/bias/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Щ
;Generator/last_layer/fully_connected/bias/Initializer/zerosFillKGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorAGenerator/last_layer/fully_connected/bias/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:
е
)Generator/last_layer/fully_connected/bias
VariableV2*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Џ
0Generator/last_layer/fully_connected/bias/AssignAssign)Generator/last_layer/fully_connected/bias;Generator/last_layer/fully_connected/bias/Initializer/zeros*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Щ
.Generator/last_layer/fully_connected/bias/readIdentity)Generator/last_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:
т
+Generator/last_layer/fully_connected/MatMulMatMul Generator/third_layer/leaky_relu0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
о
,Generator/last_layer/fully_connected/BiasAddBiasAdd+Generator/last_layer/fully_connected/MatMul.Generator/last_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
н
OGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0*
_output_shapes
:
Э
EGenerator/last_layer/batch_normalization/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *  ?
к
?Generator/last_layer/batch_normalization/gamma/Initializer/onesFillOGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorEGenerator/last_layer/batch_normalization/gamma/Initializer/ones/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:
п
.Generator/last_layer/batch_normalization/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:
Т
5Generator/last_layer/batch_normalization/gamma/AssignAssign.Generator/last_layer/batch_normalization/gamma?Generator/last_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
и
3Generator/last_layer/batch_normalization/gamma/readIdentity.Generator/last_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:
м
OGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:
Ь
EGenerator/last_layer/batch_normalization/beta/Initializer/zeros/ConstConst*
_output_shapes
: *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0
й
?Generator/last_layer/batch_normalization/beta/Initializer/zerosFillOGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorEGenerator/last_layer/batch_normalization/beta/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:
н
-Generator/last_layer/batch_normalization/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container 
П
4Generator/last_layer/batch_normalization/beta/AssignAssign-Generator/last_layer/batch_normalization/beta?Generator/last_layer/batch_normalization/beta/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(
е
2Generator/last_layer/batch_normalization/beta/readIdentity-Generator/last_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:
ъ
VGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB:*
dtype0*
_output_shapes
:
к
LGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/ConstConst*
_output_shapes
: *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB
 *    *
dtype0
ѕ
FGenerator/last_layer/batch_normalization/moving_mean/Initializer/zerosFillVGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*

index_type0*
_output_shapes	
:
ы
4Generator/last_layer/batch_normalization/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
	container *
shape:
л
;Generator/last_layer/batch_normalization/moving_mean/AssignAssign4Generator/last_layer/batch_normalization/moving_meanFGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ъ
9Generator/last_layer/batch_normalization/moving_mean/readIdentity4Generator/last_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
_output_shapes	
:
ё
YGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB:*
dtype0*
_output_shapes
:
с
OGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/ConstConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 

IGenerator/last_layer/batch_normalization/moving_variance/Initializer/onesFillYGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorOGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/Const*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*

index_type0*
_output_shapes	
:
ѓ
8Generator/last_layer/batch_normalization/moving_variance
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
	container 
ъ
?Generator/last_layer/batch_normalization/moving_variance/AssignAssign8Generator/last_layer/batch_normalization/moving_varianceIGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
validate_shape(
і
=Generator/last_layer/batch_normalization/moving_variance/readIdentity8Generator/last_layer/batch_normalization/moving_variance*
_output_shapes	
:*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance
}
8Generator/last_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
м
6Generator/last_layer/batch_normalization/batchnorm/addAdd=Generator/last_layer/batch_normalization/moving_variance/read8Generator/last_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:

8Generator/last_layer/batch_normalization/batchnorm/RsqrtRsqrt6Generator/last_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
в
6Generator/last_layer/batch_normalization/batchnorm/mulMul8Generator/last_layer/batch_normalization/batchnorm/Rsqrt3Generator/last_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
и
8Generator/last_layer/batch_normalization/batchnorm/mul_1Mul,Generator/last_layer/fully_connected/BiasAdd6Generator/last_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:џџџџџџџџџ*
T0
и
8Generator/last_layer/batch_normalization/batchnorm/mul_2Mul9Generator/last_layer/batch_normalization/moving_mean/read6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
б
6Generator/last_layer/batch_normalization/batchnorm/subSub2Generator/last_layer/batch_normalization/beta/read8Generator/last_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ф
8Generator/last_layer/batch_normalization/batchnorm/add_1Add8Generator/last_layer/batch_normalization/batchnorm/mul_16Generator/last_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:џџџџџџџџџ
j
%Generator/last_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
О
#Generator/last_layer/leaky_relu/mulMul%Generator/last_layer/leaky_relu/alpha8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
М
Generator/last_layer/leaky_reluMaximum#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
Н
<Generator/fake_image/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0
Џ
:Generator/fake_image/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zѕkН
Џ
:Generator/fake_image/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zѕk=*
dtype0*
_output_shapes
: 

DGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniformRandomUniform<Generator/fake_image/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*.
_class$
" loc:@Generator/fake_image/kernel

:Generator/fake_image/kernel/Initializer/random_uniform/subSub:Generator/fake_image/kernel/Initializer/random_uniform/max:Generator/fake_image/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
_output_shapes
: 

:Generator/fake_image/kernel/Initializer/random_uniform/mulMulDGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniform:Generator/fake_image/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*.
_class$
" loc:@Generator/fake_image/kernel

6Generator/fake_image/kernel/Initializer/random_uniformAdd:Generator/fake_image/kernel/Initializer/random_uniform/mul:Generator/fake_image/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:

У
Generator/fake_image/kernel
VariableV2*.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 

"Generator/fake_image/kernel/AssignAssignGenerator/fake_image/kernel6Generator/fake_image/kernel/Initializer/random_uniform*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:

Є
 Generator/fake_image/kernel/readIdentityGenerator/fake_image/kernel* 
_output_shapes
:
*
T0*.
_class$
" loc:@Generator/fake_image/kernel
Ј
+Generator/fake_image/bias/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0*
_output_shapes	
:
Е
Generator/fake_image/bias
VariableV2*,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
я
 Generator/fake_image/bias/AssignAssignGenerator/fake_image/bias+Generator/fake_image/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(

Generator/fake_image/bias/readIdentityGenerator/fake_image/bias*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
С
Generator/fake_image/MatMulMatMulGenerator/last_layer/leaky_relu Generator/fake_image/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
Ў
Generator/fake_image/BiasAddBiasAddGenerator/fake_image/MatMulGenerator/fake_image/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
r
Generator/fake_image/TanhTanhGenerator/fake_image/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
z
Discriminator/real_inPlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
ч
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
й
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HYН*
dtype0*
_output_shapes
: 
й
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
г
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
seed2 
о
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
: *
T0
ђ
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
ф
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
э
0Discriminator/first_layer/fully_connected/kernel
VariableV2* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
*
dtype0
й
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
у
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:

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
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:
с
0Discriminator/first_layer/fully_connected/MatMulMatMulDiscriminator/real_in5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
э
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
o
*Discriminator/first_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>
С
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
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

seed *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:

т
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
_output_shapes
: 
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
ADiscriminator/second_layer/fully_connected/bias/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0
с
/Discriminator/second_layer/fully_connected/bias
VariableV2*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0
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
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
ђ
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
№
2Discriminator/second_layer/fully_connected/BiasAddBiasAdd1Discriminator/second_layer/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
p
+Discriminator/second_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬL>
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
8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*,
_class"
 loc:@Discriminator/prob/kernel
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
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	
ќ
 Discriminator/prob/kernel/AssignAssignDiscriminator/prob/kernel4Discriminator/prob/kernel/Initializer/random_uniform*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0

Discriminator/prob/kernel/readIdentityDiscriminator/prob/kernel*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	
Ђ
)Discriminator/prob/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    
Џ
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
Discriminator/prob/bias/readIdentityDiscriminator/prob/bias**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:*
T0
Т
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ї
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
ч
2Discriminator/first_layer_1/fully_connected/MatMulMatMulGenerator/fake_image/Tanh5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
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
3Discriminator/second_layer_1/fully_connected/MatMulMatMul&Discriminator/first_layer_1/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
є
4Discriminator/second_layer_1/fully_connected/BiasAddBiasAdd3Discriminator/second_layer_1/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
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
ones_like/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
T
ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
w
	ones_likeFillones_like/Shapeones_like/Const*

index_type0*'
_output_shapes
:џџџџџџџџџ*
T0
s
logistic_loss/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
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
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/NegDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
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
logistic_loss/Log1pLog1plogistic_loss/Exp*'
_output_shapes
:џџџџџџџџџ*
T0
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*'
_output_shapes
:џџџџџџџџџ*
T0
V
ConstConst*
dtype0*
_output_shapes
:*
valueB"       
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
:џџџџџџџџџ*
T0
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
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
Ѕ
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
v
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAdd
zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
y
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*
T0*'
_output_shapes
:џџџџџџџџџ
f
logistic_loss_1/ExpExplogistic_loss_1/Select_1*
T0*'
_output_shapes
:џџџџџџџџџ
e
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*
T0*'
_output_shapes
:џџџџџџџџџ
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*
T0*'
_output_shapes
:џџџџџџџџџ
X
Const_1Const*
_output_shapes
:*
valueB"       *
dtype0
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
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
Њ
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
j
logistic_loss_2/NegNegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
Ѕ
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
w
logistic_loss_2/mulMulDiscriminator/prob_1/BiasAddones_like_1*'
_output_shapes
:џџџџџџџџџ*
T0
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
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*'
_output_shapes
:џџџџџџџџџ*
T0
t
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*
T0*'
_output_shapes
:џџџџџџџџџ
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
dtype0*
_output_shapes
: *
valueB Bgenerator_loss
_
generator_lossHistogramSummarygenerator_loss/tagMean_2*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
X
gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
dtype0
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

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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

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

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
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
gradients/Mean_1_grad/ReshapeReshape-gradients/add_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
j
gradients/Mean_1_grad/ShapeShapelogistic_loss_1*
T0*
out_type0*
_output_shapes
:
Ђ
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
l
gradients/Mean_1_grad/Shape_1Shapelogistic_loss_1*
_output_shapes
:*
T0*
out_type0
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
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0

gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
_output_shapes
:*
T0*
out_type0
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:
в
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
И
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Е
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
М
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Л
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1

5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
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
и
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
О
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
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
Т
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
z
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
_output_shapes
:*
T0*
out_type0
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
out_type0*
_output_shapes
:*
T0
о
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
к
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
о
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
_output_shapes
:*
T0
Х
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1

9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape

;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ї
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0*
_output_shapes
: 

&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:џџџџџџџџџ
Ч
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:џџџџџџџџџ*
T0
~
(gradients/logistic_loss_1/sub_grad/ShapeShapelogistic_loss_1/Select*
_output_shapes
:*
T0*
out_type0
}
*gradients/logistic_loss_1/sub_grad/Shape_1Shapelogistic_loss_1/mul*
T0*
out_type0*
_output_shapes
:
ф
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
р
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
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
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*
T0*'
_output_shapes
:џџџџџџџџџ
Э
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
э
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
я
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1

<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ*
T0
Ђ
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1

&gradients/logistic_loss/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
_output_shapes
:*
T0*
out_type0
q
(gradients/logistic_loss/mul_grad/Shape_1Shape	ones_like*
T0*
out_type0*
_output_shapes
:
о
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*'
_output_shapes
:џџџџџџџџџ*
T0
Щ
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
И
&gradients/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
Я
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ч
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1

9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape

;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1

$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*'
_output_shapes
:џџџџџџџџџ*
T0
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
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
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
t
*gradients/logistic_loss_1/mul_grad/Shape_1Shape
zeros_like*
T0*
out_type0*
_output_shapes
:
ф
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Њ
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1
zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Я
&gradients/logistic_loss_1/mul_grad/SumSum&gradients/logistic_loss_1/mul_grad/Mul8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0

&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:џџџџџџџџџ
м
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
о
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ

6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
Є
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select*'
_output_shapes
:џџџџџџџџџ
Њ
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1

2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*
T0*'
_output_shapes
:џџџџџџџџџ
ф
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
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

$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
Ё
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
ё
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:џџџџџџџџџ

5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:

:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN6^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad

Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Г
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
§
gradients/AddN_1AddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N*'
_output_shapes
:џџџџџџџџџ*
T0

7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
T0*
data_formatNHWC*
_output_shapes
:

<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_18^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad

Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select
Л
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
і
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
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

gradients/AddN_2AddNDgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:*
T0
Ѓ
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ў
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
:gradients/Discriminator/second_layer/leaky_relu_grad/zerosFill<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:џџџџџџџџџ*
T0
у
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
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
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

8gradients/Discriminator/second_layer/leaky_relu_grad/SumSum;gradients/Discriminator/second_layer/leaky_relu_grad/SelectJgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ў
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
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
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape
щ
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Ї
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
В
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
С
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0

Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosFill>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
щ
Cgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
 
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Т
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
Ф
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
ы
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape
ё
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

gradients/AddN_3AddNCgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
N*
_output_shapes
:	
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
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulMulMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ј
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
є
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
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
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1

@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Ж
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ў
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
њ
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Rgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Dgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1Reshape@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
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
Э
gradients/AddN_4AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0
Ћ
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:
Н
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
г
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
г
gradients/AddN_5AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
­
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:
С
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
й
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
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
О
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
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
[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

Т
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Ќ
Kgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
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
ч
gradients/AddN_6AddN\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ё
9gradients/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ќ
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
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
Ѕ
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
А
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
;gradients/Discriminator/first_layer_1/leaky_relu_grad/zerosFill=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:џџџџџџџџџ*
T0
ц
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
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
=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
ц
gradients/AddN_7AddN[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:

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
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ѕ
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ё
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Ogradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Agradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
ж
Hgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp@^gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeB^gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
н
Pgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeI^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
ѕ
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ*
T0

?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
Rgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeK^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape
§
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Ъ
gradients/AddN_8AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Њ
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:
Л
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
а
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ*
T0

[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradR^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
а
gradients/AddN_9AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Ќ
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
data_formatNHWC*
_output_shapes	
:*
T0
П
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ж
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1

]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Л
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0

Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/real_inYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
ь
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1

Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul

Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
П
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ђ
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1

Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ*
T0

\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1S^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

х
gradients/AddN_10AddN[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
ф
gradients/AddN_11AddNZgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:

Ё
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *fff?
В
beta1_power
VariableV2*
_output_shapes
: *
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape: *
dtype0
б
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias

beta2_power/readIdentitybeta2_power*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
э
WDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
з
MDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
љ
GDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillWDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorMDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ђ
5Discriminator/first_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:

п
<Discriminator/first_layer/fully_connected/kernel/Adam/AssignAssign5Discriminator/first_layer/fully_connected/kernel/AdamGDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
э
:Discriminator/first_layer/fully_connected/kernel/Adam/readIdentity5Discriminator/first_layer/fully_connected/kernel/Adam* 
_output_shapes
:
*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
я
YDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     
й
ODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    
џ
IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillYDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

є
7Discriminator/first_layer/fully_connected/kernel/Adam_1
VariableV2* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
*
dtype0
х
>Discriminator/first_layer/fully_connected/kernel/Adam_1/AssignAssign7Discriminator/first_layer/fully_connected/kernel/Adam_1IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ё
<Discriminator/first_layer/fully_connected/kernel/Adam_1/readIdentity7Discriminator/first_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
з
EDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ф
3Discriminator/first_layer/fully_connected/bias/Adam
VariableV2*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0
в
:Discriminator/first_layer/fully_connected/bias/Adam/AssignAssign3Discriminator/first_layer/fully_connected/bias/AdamEDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
т
8Discriminator/first_layer/fully_connected/bias/Adam/readIdentity3Discriminator/first_layer/fully_connected/bias/Adam*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:*
T0
й
GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ц
5Discriminator/first_layer/fully_connected/bias/Adam_1
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
и
<Discriminator/first_layer/fully_connected/bias/Adam_1/AssignAssign5Discriminator/first_layer/fully_connected/bias/Adam_1GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(
ц
:Discriminator/first_layer/fully_connected/bias/Adam_1/readIdentity5Discriminator/first_layer/fully_connected/bias/Adam_1*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:
я
XDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0
й
NDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    
§
HDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillXDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorNDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

є
6Discriminator/second_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:

у
=Discriminator/second_layer/fully_connected/kernel/Adam/AssignAssign6Discriminator/second_layer/fully_connected/kernel/AdamHDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
№
;Discriminator/second_layer/fully_connected/kernel/Adam/readIdentity6Discriminator/second_layer/fully_connected/kernel/Adam*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

ё
ZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
л
PDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorPDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0
і
8Discriminator/second_layer/fully_connected/kernel/Adam_1
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
щ
?Discriminator/second_layer/fully_connected/kernel/Adam_1/AssignAssign8Discriminator/second_layer/fully_connected/kernel/Adam_1JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(
є
=Discriminator/second_layer/fully_connected/kernel/Adam_1/readIdentity8Discriminator/second_layer/fully_connected/kernel/Adam_1*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

й
FDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ц
4Discriminator/second_layer/fully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:
ж
;Discriminator/second_layer/fully_connected/bias/Adam/AssignAssign4Discriminator/second_layer/fully_connected/bias/AdamFDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
х
9Discriminator/second_layer/fully_connected/bias/Adam/readIdentity4Discriminator/second_layer/fully_connected/bias/Adam*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
л
HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ш
6Discriminator/second_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:
м
=Discriminator/second_layer/fully_connected/bias/Adam_1/AssignAssign6Discriminator/second_layer/fully_connected/bias/Adam_1HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
щ
;Discriminator/second_layer/fully_connected/bias/Adam_1/readIdentity6Discriminator/second_layer/fully_connected/bias/Adam_1*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
Е
0Discriminator/prob/kernel/Adam/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Т
Discriminator/prob/kernel/Adam
VariableV2*,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name 

%Discriminator/prob/kernel/Adam/AssignAssignDiscriminator/prob/kernel/Adam0Discriminator/prob/kernel/Adam/Initializer/zeros*
_output_shapes
:	*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(
Ї
#Discriminator/prob/kernel/Adam/readIdentityDiscriminator/prob/kernel/Adam*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	
З
2Discriminator/prob/kernel/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Ф
 Discriminator/prob/kernel/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	

'Discriminator/prob/kernel/Adam_1/AssignAssign Discriminator/prob/kernel/Adam_12Discriminator/prob/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	
Ћ
%Discriminator/prob/kernel/Adam_1/readIdentity Discriminator/prob/kernel/Adam_1*
_output_shapes
:	*
T0*,
_class"
 loc:@Discriminator/prob/kernel
Ї
.Discriminator/prob/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    
Д
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
ѕ
#Discriminator/prob/bias/Adam/AssignAssignDiscriminator/prob/bias/Adam.Discriminator/prob/bias/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:

!Discriminator/prob/bias/Adam/readIdentityDiscriminator/prob/bias/Adam*
_output_shapes
:*
T0**
_class 
loc:@Discriminator/prob/bias
Љ
0Discriminator/prob/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    
Ж
Discriminator/prob/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias
ћ
%Discriminator/prob/bias/Adam_1/AssignAssignDiscriminator/prob/bias/Adam_10Discriminator/prob/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(
 
#Discriminator/prob/bias/Adam_1/readIdentityDiscriminator/prob/bias/Adam_1**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:*
T0
W
Adam/learning_rateConst*
valueB
 *ЗQ9*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wО?
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
§
FAdam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam0Discriminator/first_layer/fully_connected/kernel5Discriminator/first_layer/fully_connected/kernel/Adam7Discriminator/first_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_11* 
_output_shapes
:
*
use_locking( *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
use_nesterov( 
ю
DAdam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam.Discriminator/first_layer/fully_connected/bias3Discriminator/first_layer/fully_connected/bias/Adam5Discriminator/first_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_10*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:

GAdam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernel6Discriminator/second_layer/fully_connected/kernel/Adam8Discriminator/second_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
ђ
EAdam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/bias4Discriminator/second_layer/fully_connected/bias/Adam6Discriminator/second_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*
use_locking( *
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:

/Adam/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernelDiscriminator/prob/kernel/Adam Discriminator/prob/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
use_nesterov( *
_output_shapes
:	
љ
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

Adam/mulMulbeta1_power/read
Adam/beta1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
Й
Adam/AssignAssignbeta1_powerAdam/mul*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking( 


Adam/mul_1Mulbeta2_power/read
Adam/beta2E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
Н
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
Ў
AdamNoOp^Adam/Assign^Adam/Assign_1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
dtype0
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

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
Ј
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
n
gradients_1/Mean_2_grad/Shape_1Shapelogistic_loss_2*
_output_shapes
:*
T0*
out_type0
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
Ђ
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
І
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

gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 

gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
y
&gradients_1/logistic_loss_2_grad/ShapeShapelogistic_loss_2/sub*
out_type0*
_output_shapes
:*
T0
}
(gradients_1/logistic_loss_2_grad/Shape_1Shapelogistic_loss_2/Log1p*
T0*
out_type0*
_output_shapes
:
о
6gradients_1/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_2_grad/Shape(gradients_1/logistic_loss_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ф
$gradients_1/logistic_loss_2_grad/SumSumgradients_1/Mean_2_grad/truediv6gradients_1/logistic_loss_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
С
(gradients_1/logistic_loss_2_grad/ReshapeReshape$gradients_1/logistic_loss_2_grad/Sum&gradients_1/logistic_loss_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
&gradients_1/logistic_loss_2_grad/Sum_1Sumgradients_1/Mean_2_grad/truediv8gradients_1/logistic_loss_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ч
*gradients_1/logistic_loss_2_grad/Reshape_1Reshape&gradients_1/logistic_loss_2_grad/Sum_1(gradients_1/logistic_loss_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

1gradients_1/logistic_loss_2_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_2_grad/Reshape+^gradients_1/logistic_loss_2_grad/Reshape_1

9gradients_1/logistic_loss_2_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_2_grad/Reshape2^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

;gradients_1/logistic_loss_2_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_2_grad/Reshape_12^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

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
ъ
:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/sub_grad/Shape,gradients_1/logistic_loss_2/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ц
(gradients_1/logistic_loss_2/sub_grad/SumSum9gradients_1/logistic_loss_2_grad/tuple/control_dependency:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Э
,gradients_1/logistic_loss_2/sub_grad/ReshapeReshape(gradients_1/logistic_loss_2/sub_grad/Sum*gradients_1/logistic_loss_2/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ъ
*gradients_1/logistic_loss_2/sub_grad/Sum_1Sum9gradients_1/logistic_loss_2_grad/tuple/control_dependency<gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
~
(gradients_1/logistic_loss_2/sub_grad/NegNeg*gradients_1/logistic_loss_2/sub_grad/Sum_1*
T0*
_output_shapes
:
б
.gradients_1/logistic_loss_2/sub_grad/Reshape_1Reshape(gradients_1/logistic_loss_2/sub_grad/Neg,gradients_1/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5gradients_1/logistic_loss_2/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/sub_grad/Reshape/^gradients_1/logistic_loss_2/sub_grad/Reshape_1
Ђ
=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/sub_grad/Reshape6^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ј
?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/sub_grad/Reshape_16^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Џ
,gradients_1/logistic_loss_2/Log1p_grad/add/xConst<^gradients_1/logistic_loss_2_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0*
_output_shapes
: 
І
*gradients_1/logistic_loss_2/Log1p_grad/addAdd,gradients_1/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*'
_output_shapes
:џџџџџџџџџ*
T0

1gradients_1/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_2/Log1p_grad/add*
T0*'
_output_shapes
:џџџџџџџџџ
г
*gradients_1/logistic_loss_2/Log1p_grad/mulMul;gradients_1/logistic_loss_2_grad/tuple/control_dependency_11gradients_1/logistic_loss_2/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ

2gradients_1/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
ћ
.gradients_1/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_2/Select_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
§
0gradients_1/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients_1/logistic_loss_2/Select_grad/zeros_like=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
Є
8gradients_1/logistic_loss_2/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_2/Select_grad/Select1^gradients_1/logistic_loss_2/Select_grad/Select_1
Ќ
@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_2/Select_grad/Select9^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select
В
Bgradients_1/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_2/Select_grad/Select_19^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_grad/Select_1*'
_output_shapes
:џџџџџџџџџ

*gradients_1/logistic_loss_2/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
_output_shapes
:*
T0*
out_type0
w
,gradients_1/logistic_loss_2/mul_grad/Shape_1Shapeones_like_1*
T0*
out_type0*
_output_shapes
:
ъ
:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/mul_grad/Shape,gradients_1/logistic_loss_2/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Џ
(gradients_1/logistic_loss_2/mul_grad/MulMul?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1ones_like_1*
T0*'
_output_shapes
:џџџџџџџџџ
е
(gradients_1/logistic_loss_2/mul_grad/SumSum(gradients_1/logistic_loss_2/mul_grad/Mul:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Э
,gradients_1/logistic_loss_2/mul_grad/ReshapeReshape(gradients_1/logistic_loss_2/mul_grad/Sum*gradients_1/logistic_loss_2/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Т
*gradients_1/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1*'
_output_shapes
:џџџџџџџџџ*
T0
л
*gradients_1/logistic_loss_2/mul_grad/Sum_1Sum*gradients_1/logistic_loss_2/mul_grad/Mul_1<gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
г
.gradients_1/logistic_loss_2/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_2/mul_grad/Sum_1,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5gradients_1/logistic_loss_2/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/mul_grad/Reshape/^gradients_1/logistic_loss_2/mul_grad/Reshape_1
Ђ
=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/mul_grad/Reshape6^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/mul_grad/Reshape
Ј
?gradients_1/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/mul_grad/Reshape_16^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ђ
(gradients_1/logistic_loss_2/Exp_grad/mulMul*gradients_1/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

4gradients_1/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
0gradients_1/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual(gradients_1/logistic_loss_2/Exp_grad/mul4gradients_1/logistic_loss_2/Select_1_grad/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
ь
2gradients_1/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual4gradients_1/logistic_loss_2/Select_1_grad/zeros_like(gradients_1/logistic_loss_2/Exp_grad/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Њ
:gradients_1/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_2/Select_1_grad/Select3^gradients_1/logistic_loss_2/Select_1_grad/Select_1
Д
Bgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_2/Select_1_grad/Select;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_1_grad/Select*'
_output_shapes
:џџџџџџџџџ
К
Dgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_2/Select_1_grad/Select_1;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/logistic_loss_2/Select_1_grad/Select_1*'
_output_shapes
:џџџџџџџџџ
Ѕ
(gradients_1/logistic_loss_2/Neg_grad/NegNegBgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0

gradients_1/AddNAddN@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_2/Neg_grad/Neg*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*
N*'
_output_shapes
:џџџџџџџџџ

9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
data_formatNHWC*
_output_shapes
:*
T0

>gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN:^gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad

Fgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
У
Hgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ў
3gradients_1/Discriminator/prob_1/MatMul_grad/MatMulMatMulFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0

5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(
Г
=gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul6^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1
С
Egradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Discriminator/prob_1/MatMul_grad/MatMul>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul
О
Ggradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	
Љ
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
Д
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Х
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosFill@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
ы
Egradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
І
Ngradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ъ
?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
Ь
Agradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SumSum?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectNgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1SumAgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1Pgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Bgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
й
Igradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOpA^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeC^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
ѓ
Qgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeJ^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape
љ
Sgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityBgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1J^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ*
T0

Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
И
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
В
Rgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulRgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ў
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Ѓ
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Tgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Fgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
х
Mgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpE^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeG^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
ё
Ugradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeN^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 

Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityFgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1N^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
л
gradients_1/AddN_1AddNSgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:џџџџџџџџџ*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N
Б
Qgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:
Ч
Vgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1R^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
с
^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1W^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Є
`gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityQgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ц
Kgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
А
Mgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ћ
Ugradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpL^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulN^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
Ё
]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityKgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulV^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

_gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityMgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1V^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*`
_classV
TRloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
Ї
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
В
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
м
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zerosFill?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
ш
Dgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Ѓ
Mgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
п
>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
с
@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SumSum>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectMgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1Ogradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Agradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
ж
Hgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp@^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeB^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
я
Pgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeI^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape
ѕ
Rgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityAgradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1I^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1

Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ж
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Џ
Qgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulQgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ћ
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
 
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Sgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Egradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
т
Lgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpD^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeF^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
э
Tgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeM^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 

Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityEgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1M^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
и
gradients_1/AddN_2AddNRgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
А
Pgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
data_formatNHWC*
_output_shapes	
:*
T0
Х
Ugradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2Q^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
о
]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2V^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
 
_gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradV^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
У
Jgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
Ё
Lgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ј
Tgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpK^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulM^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1

\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityJgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulU^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityLgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1U^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

ы
3gradients_1/Generator/fake_image/Tanh_grad/TanhGradTanhGradGenerator/fake_image/Tanh\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
К
9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0
И
>gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad4^gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
У
Fgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
Ф
Hgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*L
_classB
@>loc:@gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad

3gradients_1/Generator/fake_image/MatMul_grad/MatMulMatMulFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency Generator/fake_image/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
љ
5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1MatMulGenerator/last_layer/leaky_reluFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
Г
=gradients_1/Generator/fake_image/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Generator/fake_image/MatMul_grad/MatMul6^gradients_1/Generator/fake_image/MatMul_grad/MatMul_1
С
Egradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/MatMul_grad/MatMul>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
П
Ggradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul_1* 
_output_shapes
:


6gradients_1/Generator/last_layer/leaky_relu_grad/ShapeShape#Generator/last_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
А
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Н
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2ShapeEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0

<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ћ
6gradients_1/Generator/last_layer/leaky_relu_grad/zerosFill8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
п
=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

Fgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Generator/last_layer/leaky_relu_grad/Shape8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
В
7gradients_1/Generator/last_layer/leaky_relu_grad/SelectSelect=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency6gradients_1/Generator/last_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
Д
9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Select=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqual6gradients_1/Generator/last_layer/leaky_relu_grad/zerosEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
ќ
4gradients_1/Generator/last_layer/leaky_relu_grad/SumSum7gradients_1/Generator/last_layer/leaky_relu_grad/SelectFgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ђ
8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Generator/last_layer/leaky_relu_grad/Sum6gradients_1/Generator/last_layer/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0

6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Hgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ј
:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_18gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
С
Agradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape;^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1
г
Igradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeB^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
й
Kgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1B^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
}
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Д
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0

Jgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ї
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulMulIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0

8gradients_1/Generator/last_layer/leaky_relu/mul_grad/SumSum8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulJgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ь
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ц
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Mul%Generator/last_layer/leaky_relu/alphaIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Э
Egradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
б
Mgradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*O
_classE
CAloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
щ
Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
У
gradients_1/AddN_3AddNKgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0
Ч
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Generator/last_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:

Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
й
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_3_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Н
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_3agradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
З
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape
А
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Л
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Generator/last_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
й
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѓ
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Generator/last_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:џџџџџџџџџ*
T0
Ф
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Н
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Generator/last_layer/fully_connected/BiasAddbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Ъ
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
З
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
А
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
о
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg
Л
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:*
T0

bgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg
љ
Igradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
_output_shapes	
:*
T0*
data_formatNHWC

Ngradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad
А
Vgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ

Xgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Generator/last_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Generator/last_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Ђ
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul
Ј
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*b
_classX
VTloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:*
T0
А
Cgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Generator/last_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0

Egradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/third_layer/leaky_reluVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
у
Mgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1

Ugradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*V
_classL
JHloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ*
T0
џ
Wgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

§
gradients_1/AddN_4AddNdgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
С
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_43Generator/last_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
Ш
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_48Generator/last_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:*
T0
ў
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1

`gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:*
T0
 
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:

7gradients_1/Generator/third_layer/leaky_relu_grad/ShapeShape$Generator/third_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
В
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Ю
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ў
7gradients_1/Generator/third_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:џџџџџџџџџ*
T0
т
>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

Ggradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/third_layer/leaky_relu_grad/Shape9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Х
8gradients_1/Generator/third_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/third_layer/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
Ч
:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/third_layer/leaky_relu_grad/zerosUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
џ
5gradients_1/Generator/third_layer/leaky_relu_grad/SumSum8gradients_1/Generator/third_layer/leaky_relu_grad/SelectGgradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ѕ
9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/third_layer/leaky_relu_grad/Sum7gradients_1/Generator/third_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Igradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ћ
;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_19gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ф
Bgradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
з
Jgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*L
_classB
@>loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape
н
Lgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
~
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ж
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

Kgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
њ
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0

9gradients_1/Generator/third_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
я
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
щ
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/third_layer/leaky_relu/alphaJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
а
Fgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
е
Ngradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
э
Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Ц
gradients_1/AddN_5AddNLgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
Щ
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape9Generator/third_layer/batch_normalization/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0

Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
м
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_5`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Р
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_5bgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Й
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
Л
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
Д
egradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:*
T0
Н
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape-Generator/third_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:

Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
м
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
І
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency7Generator/third_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:џџџџџџџџџ*
T0
Ч
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumSumNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Р
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul-Generator/third_layer/fully_connected/BiasAddcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Э
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1bgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Й
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
Л
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
Д
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
р
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegNegegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpf^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1M^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg
П
agradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
 
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:*
T0
ћ
Jgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradcgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:*
T0

Ogradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpd^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyK^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Д
Wgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitycgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ

Ygradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*]
_classS
QOloc:@gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad

Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_17Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1:Generator/third_layer/batch_normalization/moving_mean/read*
_output_shapes	
:*
T0

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulQ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
І
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
Ќ
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
Г
Dgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/third_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0

Fgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1MatMul!Generator/second_layer/leaky_reluWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ц
Ngradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1

Vgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

Xgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:


gradients_1/AddN_6AddNegradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
У
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_64Generator/third_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
Ъ
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_69Generator/third_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:

Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpM^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1

agradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:
Є
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1

8gradients_1/Generator/second_layer/leaky_relu_grad/ShapeShape%Generator/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Д
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
а
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2ShapeVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

8gradients_1/Generator/second_layer/leaky_relu_grad/zerosFill:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
х
?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

Hgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/Generator/second_layer/leaky_relu_grad/Shape:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Щ
9gradients_1/Generator/second_layer/leaky_relu_grad/SelectSelect?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency8gradients_1/Generator/second_layer/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
Ы
;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Select?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqual8gradients_1/Generator/second_layer/leaky_relu_grad/zerosVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

6gradients_1/Generator/second_layer/leaky_relu_grad/SumSum9gradients_1/Generator/second_layer/leaky_relu_grad/SelectHgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ј
:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeReshape6gradients_1/Generator/second_layer/leaky_relu_grad/Sum8gradients_1/Generator/second_layer/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0

8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1Sum;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Jgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ў
<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1Reshape8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ч
Cgradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_depsNoOp;^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape=^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1
л
Kgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeD^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
с
Mgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1D^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
И
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
 
Lgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
§
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulMulKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0

:gradients_1/Generator/second_layer/leaky_relu/mul_grad/SumSum:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulLgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ђ
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeReshape:gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ь
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Mul'Generator/second_layer/leaky_relu/alphaKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Ngradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Ggradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp?^gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeA^gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
й
Ogradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeH^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
ё
Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1H^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Щ
gradients_1/AddN_7AddNMgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Ы
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape:Generator/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:

Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
п
agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_7agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_7cgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
М
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
П
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
И
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
П
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape.Generator/second_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
п
agradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Љ
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
Ъ
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mulagradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
У
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ё
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul.Generator/second_layer/fully_connected/BiasAdddgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
а
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1cgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
М
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
Tshape0*
_output_shapes	
:*
T0

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
П
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
И
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
т
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/NegNegfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpg^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1N^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
У
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Є
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:
§
Kgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGraddgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Pgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpe^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyL^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
И
Xgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitydgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ

Zgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_18Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
Ё
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Muldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1;Generator/second_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulR^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Њ
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
А
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
Ж
Egradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulMatMulXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency2Generator/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 

Ggradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/first_layer/leaky_reluXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
щ
Ogradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpF^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulH^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1

Wgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityEgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulP^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

Ygradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityGgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1P^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*Z
_classP
NLloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1

gradients_1/AddN_8AddNfgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
Х
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_85Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
Ь
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_8:Generator/second_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:*
T0

Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpN^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Ђ
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul
Ј
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:

7gradients_1/Generator/first_layer/leaky_relu_grad/ShapeShape$Generator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
І
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
а
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2ShapeWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ў
7gradients_1/Generator/first_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
ж
>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

Ggradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/first_layer/leaky_relu_grad/Shape9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
8gradients_1/Generator/first_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
Щ
:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/first_layer/leaky_relu_grad/zerosWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
џ
5gradients_1/Generator/first_layer/leaky_relu_grad/SumSum8gradients_1/Generator/first_layer/leaky_relu_grad/SelectGgradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ѕ
9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/first_layer/leaky_relu_grad/Sum7gradients_1/Generator/first_layer/leaky_relu_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Igradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ћ
;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_19gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ф
Bgradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
з
Jgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*L
_classB
@>loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape
н
Lgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
~
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Њ
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Kgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ю
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients_1/Generator/first_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
я
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
щ
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/first_layer/leaky_relu/alphaJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
а
Fgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
е
Ngradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*P
_classF
DBloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape
э
Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*R
_classH
FDloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
Ц
gradients_1/AddN_9AddNLgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Њ
Jgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:
Й
Ogradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_9K^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ь
Wgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_9P^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

Ygradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*]
_classS
QOloc:@gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
В
Dgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/first_layer/fully_connected/kernel/read*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(*
T0

Fgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise_inWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d*
transpose_a(*
transpose_b( *
T0
ц
Ngradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1

Vgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџd

Xgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
_output_shapes
:	d*
T0*Y
_classO
MKloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1

beta1_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 

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
Т
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
|
beta1_power_1/readIdentitybeta1_power_1*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: *
T0

beta2_power_1/initial_valueConst*
_output_shapes
: *,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *wО?*
dtype0

beta2_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape: 
Т
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias
|
beta2_power_1/readIdentitybeta2_power_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
х
SGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
Я
IGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    
ш
CGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d*
T0
ш
1Generator/first_layer/fully_connected/kernel/Adam
VariableV2*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name 
Ю
8Generator/first_layer/fully_connected/kernel/Adam/AssignAssign1Generator/first_layer/fully_connected/kernel/AdamCGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d
р
6Generator/first_layer/fully_connected/kernel/Adam/readIdentity1Generator/first_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
ч
UGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0
б
KGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ю
EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	d*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0
ъ
3Generator/first_layer/fully_connected/kernel/Adam_1
VariableV2*
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container 
д
:Generator/first_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/first_layer/fully_connected/kernel/Adam_1EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d
ф
8Generator/first_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/first_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
Я
AGenerator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
м
/Generator/first_layer/fully_connected/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container 
Т
6Generator/first_layer/fully_connected/bias/Adam/AssignAssign/Generator/first_layer/fully_connected/bias/AdamAGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ж
4Generator/first_layer/fully_connected/bias/Adam/readIdentity/Generator/first_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:
б
CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
о
1Generator/first_layer/fully_connected/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0
Ш
8Generator/first_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/first_layer/fully_connected/bias/Adam_1CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
к
6Generator/first_layer/fully_connected/bias/Adam_1/readIdentity1Generator/first_layer/fully_connected/bias/Adam_1*
_output_shapes	
:*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
ч
TGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
б
JGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
э
DGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillTGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorJGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ь
2Generator/second_layer/fully_connected/kernel/Adam
VariableV2*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
г
9Generator/second_layer/fully_connected/kernel/Adam/AssignAssign2Generator/second_layer/fully_connected/kernel/AdamDGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

ф
7Generator/second_layer/fully_connected/kernel/Adam/readIdentity2Generator/second_layer/fully_connected/kernel/Adam*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

щ
VGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
г
LGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ѓ
FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillVGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ю
4Generator/second_layer/fully_connected/kernel/Adam_1
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container 
й
;Generator/second_layer/fully_connected/kernel/Adam_1/AssignAssign4Generator/second_layer/fully_connected/kernel/Adam_1FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

ш
9Generator/second_layer/fully_connected/kernel/Adam_1/readIdentity4Generator/second_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
б
BGenerator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
о
0Generator/second_layer/fully_connected/bias/Adam
VariableV2*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ц
7Generator/second_layer/fully_connected/bias/Adam/AssignAssign0Generator/second_layer/fully_connected/bias/AdamBGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
й
5Generator/second_layer/fully_connected/bias/Adam/readIdentity0Generator/second_layer/fully_connected/bias/Adam*
_output_shapes	
:*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias
г
DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
р
2Generator/second_layer/fully_connected/bias/Adam_1
VariableV2*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ь
9Generator/second_layer/fully_connected/bias/Adam_1/AssignAssign2Generator/second_layer/fully_connected/bias/Adam_1DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
н
7Generator/second_layer/fully_connected/bias/Adam_1/readIdentity2Generator/second_layer/fully_connected/bias/Adam_1*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:
л
GGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ш
5Generator/second_layer/batch_normalization/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:
к
<Generator/second_layer/batch_normalization/gamma/Adam/AssignAssign5Generator/second_layer/batch_normalization/gamma/AdamGGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ш
:Generator/second_layer/batch_normalization/gamma/Adam/readIdentity5Generator/second_layer/batch_normalization/gamma/Adam*
_output_shapes	
:*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
н
IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*
_output_shapes	
:*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*    *
dtype0
ъ
7Generator/second_layer/batch_normalization/gamma/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container 
р
>Generator/second_layer/batch_normalization/gamma/Adam_1/AssignAssign7Generator/second_layer/batch_normalization/gamma/Adam_1IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(
ь
<Generator/second_layer/batch_normalization/gamma/Adam_1/readIdentity7Generator/second_layer/batch_normalization/gamma/Adam_1*
_output_shapes	
:*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
й
FGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ц
4Generator/second_layer/batch_normalization/beta/Adam
VariableV2*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ж
;Generator/second_layer/batch_normalization/beta/Adam/AssignAssign4Generator/second_layer/batch_normalization/beta/AdamFGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(
х
9Generator/second_layer/batch_normalization/beta/Adam/readIdentity4Generator/second_layer/batch_normalization/beta/Adam*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:
л
HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ш
6Generator/second_layer/batch_normalization/beta/Adam_1
VariableV2*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
м
=Generator/second_layer/batch_normalization/beta/Adam_1/AssignAssign6Generator/second_layer/batch_normalization/beta/Adam_1HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
щ
;Generator/second_layer/batch_normalization/beta/Adam_1/readIdentity6Generator/second_layer/batch_normalization/beta/Adam_1*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:
х
SGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      
Я
IGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
щ
CGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ъ
1Generator/third_layer/fully_connected/kernel/Adam
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
Я
8Generator/third_layer/fully_connected/kernel/Adam/AssignAssign1Generator/third_layer/fully_connected/kernel/AdamCGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

с
6Generator/third_layer/fully_connected/kernel/Adam/readIdentity1Generator/third_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

ч
UGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
б
KGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
я
EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ь
3Generator/third_layer/fully_connected/kernel/Adam_1
VariableV2* 
_output_shapes
:
*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
*
dtype0
е
:Generator/third_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/third_layer/fully_connected/kernel/Adam_1EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

х
8Generator/third_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/third_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

Я
AGenerator/third_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
м
/Generator/third_layer/fully_connected/bias/Adam
VariableV2*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:*
dtype0
Т
6Generator/third_layer/fully_connected/bias/Adam/AssignAssign/Generator/third_layer/fully_connected/bias/AdamAGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ж
4Generator/third_layer/fully_connected/bias/Adam/readIdentity/Generator/third_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:
б
CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0
о
1Generator/third_layer/fully_connected/bias/Adam_1
VariableV2*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ш
8Generator/third_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/third_layer/fully_connected/bias/Adam_1CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
к
6Generator/third_layer/fully_connected/bias/Adam_1/readIdentity1Generator/third_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:
й
FGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*    *
dtype0
ц
4Generator/third_layer/batch_normalization/gamma/Adam
VariableV2*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
ж
;Generator/third_layer/batch_normalization/gamma/Adam/AssignAssign4Generator/third_layer/batch_normalization/gamma/AdamFGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
х
9Generator/third_layer/batch_normalization/gamma/Adam/readIdentity4Generator/third_layer/batch_normalization/gamma/Adam*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
л
HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ш
6Generator/third_layer/batch_normalization/gamma/Adam_1
VariableV2*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
м
=Generator/third_layer/batch_normalization/gamma/Adam_1/AssignAssign6Generator/third_layer/batch_normalization/gamma/Adam_1HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma
щ
;Generator/third_layer/batch_normalization/gamma/Adam_1/readIdentity6Generator/third_layer/batch_normalization/gamma/Adam_1*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
з
EGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    
ф
3Generator/third_layer/batch_normalization/beta/Adam
VariableV2*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
в
:Generator/third_layer/batch_normalization/beta/Adam/AssignAssign3Generator/third_layer/batch_normalization/beta/AdamEGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
т
8Generator/third_layer/batch_normalization/beta/Adam/readIdentity3Generator/third_layer/batch_normalization/beta/Adam*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:
й
GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ц
5Generator/third_layer/batch_normalization/beta/Adam_1
VariableV2*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
и
<Generator/third_layer/batch_normalization/beta/Adam_1/AssignAssign5Generator/third_layer/batch_normalization/beta/Adam_1GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ц
:Generator/third_layer/batch_normalization/beta/Adam_1/readIdentity5Generator/third_layer/batch_normalization/beta/Adam_1*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:
у
RGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      
Э
HGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
х
BGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zerosFillRGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ш
0Generator/last_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:

Ы
7Generator/last_layer/fully_connected/kernel/Adam/AssignAssign0Generator/last_layer/fully_connected/kernel/AdamBGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
о
5Generator/last_layer/fully_connected/kernel/Adam/readIdentity0Generator/last_layer/fully_connected/kernel/Adam* 
_output_shapes
:
*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
х
TGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Я
JGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ы
DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillTGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorJGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ъ
2Generator/last_layer/fully_connected/kernel/Adam_1
VariableV2*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

б
9Generator/last_layer/fully_connected/kernel/Adam_1/AssignAssign2Generator/last_layer/fully_connected/kernel/Adam_1DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
т
7Generator/last_layer/fully_connected/kernel/Adam_1/readIdentity2Generator/last_layer/fully_connected/kernel/Adam_1*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

й
PGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
Щ
FGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    
и
@Generator/last_layer/fully_connected/bias/Adam/Initializer/zerosFillPGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorFGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/Const*
_output_shapes	
:*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0
к
.Generator/last_layer/fully_connected/bias/Adam
VariableV2*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
О
5Generator/last_layer/fully_connected/bias/Adam/AssignAssign.Generator/last_layer/fully_connected/bias/Adam@Generator/last_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
г
3Generator/last_layer/fully_connected/bias/Adam/readIdentity.Generator/last_layer/fully_connected/bias/Adam*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:
л
RGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
Ы
HGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
о
BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zerosFillRGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:
м
0Generator/last_layer/fully_connected/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
Ф
7Generator/last_layer/fully_connected/bias/Adam_1/AssignAssign0Generator/last_layer/fully_connected/bias/Adam_1BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
з
5Generator/last_layer/fully_connected/bias/Adam_1/readIdentity0Generator/last_layer/fully_connected/bias/Adam_1*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:
у
UGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0*
_output_shapes
:
г
KGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ь
EGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zerosFillUGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorKGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:
ф
3Generator/last_layer/batch_normalization/gamma/Adam
VariableV2*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
в
:Generator/last_layer/batch_normalization/gamma/Adam/AssignAssign3Generator/last_layer/batch_normalization/gamma/AdamEGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
т
8Generator/last_layer/batch_normalization/gamma/Adam/readIdentity3Generator/last_layer/batch_normalization/gamma/Adam*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:
х
WGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0*
_output_shapes
:
е
MGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ђ
GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zerosFillWGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorMGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:
ц
5Generator/last_layer/batch_normalization/gamma/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma
и
<Generator/last_layer/batch_normalization/gamma/Adam_1/AssignAssign5Generator/last_layer/batch_normalization/gamma/Adam_1GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
ц
:Generator/last_layer/batch_normalization/gamma/Adam_1/readIdentity5Generator/last_layer/batch_normalization/gamma/Adam_1*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:*
T0
с
TGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:*
dtype0*
_output_shapes
:
б
JGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ш
DGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zerosFillTGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorJGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:
т
2Generator/last_layer/batch_normalization/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:
Ю
9Generator/last_layer/batch_normalization/beta/Adam/AssignAssign2Generator/last_layer/batch_normalization/beta/AdamDGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
п
7Generator/last_layer/batch_normalization/beta/Adam/readIdentity2Generator/last_layer/batch_normalization/beta/Adam*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:
у
VGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:*
dtype0*
_output_shapes
:
г
LGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ю
FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zerosFillVGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/Const*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:*
T0
ф
4Generator/last_layer/batch_normalization/beta/Adam_1
VariableV2*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
д
;Generator/last_layer/batch_normalization/beta/Adam_1/AssignAssign4Generator/last_layer/batch_normalization/beta/Adam_1FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
у
9Generator/last_layer/batch_normalization/beta/Adam_1/readIdentity4Generator/last_layer/batch_normalization/beta/Adam_1*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:
У
BGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
­
8Generator/fake_image/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѕ
2Generator/fake_image/kernel/Adam/Initializer/zerosFillBGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensor8Generator/fake_image/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:

Ш
 Generator/fake_image/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:


'Generator/fake_image/kernel/Adam/AssignAssign Generator/fake_image/kernel/Adam2Generator/fake_image/kernel/Adam/Initializer/zeros*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ў
%Generator/fake_image/kernel/Adam/readIdentity Generator/fake_image/kernel/Adam* 
_output_shapes
:
*
T0*.
_class$
" loc:@Generator/fake_image/kernel
Х
DGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
Џ
:Generator/fake_image/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ћ
4Generator/fake_image/kernel/Adam_1/Initializer/zerosFillDGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensor:Generator/fake_image/kernel/Adam_1/Initializer/zeros/Const*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:

Ъ
"Generator/fake_image/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:


)Generator/fake_image/kernel/Adam_1/AssignAssign"Generator/fake_image/kernel/Adam_14Generator/fake_image/kernel/Adam_1/Initializer/zeros*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
В
'Generator/fake_image/kernel/Adam_1/readIdentity"Generator/fake_image/kernel/Adam_1*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:

­
0Generator/fake_image/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0*
_output_shapes	
:
К
Generator/fake_image/bias/Adam
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ў
%Generator/fake_image/bias/Adam/AssignAssignGenerator/fake_image/bias/Adam0Generator/fake_image/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias
Ѓ
#Generator/fake_image/bias/Adam/readIdentityGenerator/fake_image/bias/Adam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
Џ
2Generator/fake_image/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0*
_output_shapes	
:
М
 Generator/fake_image/bias/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:*
dtype0*
_output_shapes	
:

'Generator/fake_image/bias/Adam_1/AssignAssign Generator/fake_image/bias/Adam_12Generator/fake_image/bias/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ї
%Generator/fake_image/bias/Adam_1/readIdentity Generator/fake_image/bias/Adam_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
Y
Adam_1/learning_rateConst*
valueB
 *ЗQ9*
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
 *wО?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
Н
DAdam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/first_layer/fully_connected/kernel1Generator/first_layer/fully_connected/kernel/Adam3Generator/first_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	d
А
BAdam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/first_layer/fully_connected/bias/Generator/first_layer/fully_connected/bias/Adam1Generator/first_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
Ф
EAdam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam-Generator/second_layer/fully_connected/kernel2Generator/second_layer/fully_connected/kernel/Adam4Generator/second_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
Ж
CAdam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam+Generator/second_layer/fully_connected/bias0Generator/second_layer/fully_connected/bias/Adam2Generator/second_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:
й
HAdam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam0Generator/second_layer/batch_normalization/gamma5Generator/second_layer/batch_normalization/gamma/Adam7Generator/second_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
в
GAdam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam/Generator/second_layer/batch_normalization/beta4Generator/second_layer/batch_normalization/beta/Adam6Generator/second_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:
О
DAdam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/third_layer/fully_connected/kernel1Generator/third_layer/fully_connected/kernel/Adam3Generator/third_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
*
use_locking( *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
use_nesterov( 
А
BAdam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/third_layer/fully_connected/bias/Generator/third_layer/fully_connected/bias/Adam1Generator/third_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
г
GAdam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam/Generator/third_layer/batch_normalization/gamma4Generator/third_layer/batch_normalization/gamma/Adam6Generator/third_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
Ь
FAdam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdam	ApplyAdam.Generator/third_layer/batch_normalization/beta3Generator/third_layer/batch_normalization/beta/Adam5Generator/third_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:
И
CAdam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Generator/last_layer/fully_connected/kernel0Generator/last_layer/fully_connected/kernel/Adam2Generator/last_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
Њ
AAdam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Generator/last_layer/fully_connected/bias.Generator/last_layer/fully_connected/bias/Adam0Generator/last_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Э
FAdam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Generator/last_layer/batch_normalization/gamma3Generator/last_layer/batch_normalization/gamma/Adam5Generator/last_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:
Ц
EAdam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Generator/last_layer/batch_normalization/beta2Generator/last_layer/batch_normalization/beta/Adam4Generator/last_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
и
3Adam_1/update_Generator/fake_image/kernel/ApplyAdam	ApplyAdamGenerator/fake_image/kernel Generator/fake_image/kernel/Adam"Generator/fake_image/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
*
use_locking( *
T0*.
_class$
" loc:@Generator/fake_image/kernel*
use_nesterov( 
Ъ
1Adam_1/update_Generator/fake_image/bias/ApplyAdam	ApplyAdamGenerator/fake_image/biasGenerator/fake_image/bias/Adam Generator/fake_image/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
е	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*,
_class"
 loc:@Generator/fake_image/bias
Њ
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
з	
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta22^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: *
T0
Ў
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: "ыіэжl     ДC	lQъ§жAJЩй
пН
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
Ttype*1.12.02v1.12.0-0-ga6d8ffae09і
u
Generator/noise_inPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџd*
shape:џџџџџџџџџd
п
MGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
б
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&О*
dtype0*
_output_shapes
: 
б
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&>*
dtype0
Ц
UGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d*

seed *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
seed2 
Ю
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
: 
с
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
_output_shapes
:	d*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
г
GGenerator/first_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
у
,Generator/first_layer/fully_connected/kernel
VariableV2*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d
Ш
3Generator/first_layer/fully_connected/kernel/AssignAssign,Generator/first_layer/fully_connected/kernelGGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d
ж
1Generator/first_layer/fully_connected/kernel/readIdentity,Generator/first_layer/fully_connected/kernel*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d*
T0
Ъ
<Generator/first_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
з
*Generator/first_layer/fully_connected/bias
VariableV2*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Г
1Generator/first_layer/fully_connected/bias/AssignAssign*Generator/first_layer/fully_connected/bias<Generator/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ь
/Generator/first_layer/fully_connected/bias/readIdentity*Generator/first_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:
ж
,Generator/first_layer/fully_connected/MatMulMatMulGenerator/noise_in1Generator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
с
-Generator/first_layer/fully_connected/BiasAddBiasAdd,Generator/first_layer/fully_connected/MatMul/Generator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
k
&Generator/first_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Е
$Generator/first_layer/leaky_relu/mulMul&Generator/first_layer/leaky_relu/alpha-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Г
 Generator/first_layer/leaky_reluMaximum$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
с
NGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0
г
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   О
г
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
Ъ
VGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformNGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*

seed *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
seed2 *
dtype0
в
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
_output_shapes
: 
ц
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulVGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

и
HGenerator/second_layer/fully_connected/kernel/Initializer/random_uniformAddLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

ч
-Generator/second_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:

Э
4Generator/second_layer/fully_connected/kernel/AssignAssign-Generator/second_layer/fully_connected/kernelHGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

к
2Generator/second_layer/fully_connected/kernel/readIdentity-Generator/second_layer/fully_connected/kernel*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

Ь
=Generator/second_layer/fully_connected/bias/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
й
+Generator/second_layer/fully_connected/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias
З
2Generator/second_layer/fully_connected/bias/AssignAssign+Generator/second_layer/fully_connected/bias=Generator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Я
0Generator/second_layer/fully_connected/bias/readIdentity+Generator/second_layer/fully_connected/bias*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:
ц
-Generator/second_layer/fully_connected/MatMulMatMul Generator/first_layer/leaky_relu2Generator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
ф
.Generator/second_layer/fully_connected/BiasAddBiasAdd-Generator/second_layer/fully_connected/MatMul0Generator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
е
AGenerator/second_layer/batch_normalization/gamma/Initializer/onesConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
у
0Generator/second_layer/batch_normalization/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:
Ъ
7Generator/second_layer/batch_normalization/gamma/AssignAssign0Generator/second_layer/batch_normalization/gammaAGenerator/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
о
5Generator/second_layer/batch_normalization/gamma/readIdentity0Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
д
AGenerator/second_layer/batch_normalization/beta/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
с
/Generator/second_layer/batch_normalization/beta
VariableV2*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
Ч
6Generator/second_layer/batch_normalization/beta/AssignAssign/Generator/second_layer/batch_normalization/betaAGenerator/second_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
л
4Generator/second_layer/batch_normalization/beta/readIdentity/Generator/second_layer/batch_normalization/beta*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:
т
HGenerator/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*
_output_shapes	
:*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
valueB*    *
dtype0
я
6Generator/second_layer/batch_normalization/moving_mean
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
	container *
shape:
у
=Generator/second_layer/batch_normalization/moving_mean/AssignAssign6Generator/second_layer/batch_normalization/moving_meanHGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
№
;Generator/second_layer/batch_normalization/moving_mean/readIdentity6Generator/second_layer/batch_normalization/moving_mean*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
_output_shapes	
:
щ
KGenerator/second_layer/batch_normalization/moving_variance/Initializer/onesConst*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ї
:Generator/second_layer/batch_normalization/moving_variance
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance
ђ
AGenerator/second_layer/batch_normalization/moving_variance/AssignAssign:Generator/second_layer/batch_normalization/moving_varianceKGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ќ
?Generator/second_layer/batch_normalization/moving_variance/readIdentity:Generator/second_layer/batch_normalization/moving_variance*
_output_shapes	
:*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance

:Generator/second_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
т
8Generator/second_layer/batch_normalization/batchnorm/addAdd?Generator/second_layer/batch_normalization/moving_variance/read:Generator/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:
Ѓ
:Generator/second_layer/batch_normalization/batchnorm/RsqrtRsqrt8Generator/second_layer/batch_normalization/batchnorm/add*
_output_shapes	
:*
T0
и
8Generator/second_layer/batch_normalization/batchnorm/mulMul:Generator/second_layer/batch_normalization/batchnorm/Rsqrt5Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
о
:Generator/second_layer/batch_normalization/batchnorm/mul_1Mul.Generator/second_layer/fully_connected/BiasAdd8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
о
:Generator/second_layer/batch_normalization/batchnorm/mul_2Mul;Generator/second_layer/batch_normalization/moving_mean/read8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
з
8Generator/second_layer/batch_normalization/batchnorm/subSub4Generator/second_layer/batch_normalization/beta/read:Generator/second_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:*
T0
ъ
:Generator/second_layer/batch_normalization/batchnorm/add_1Add:Generator/second_layer/batch_normalization/batchnorm/mul_18Generator/second_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:џџџџџџџџџ
l
'Generator/second_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ф
%Generator/second_layer/leaky_relu/mulMul'Generator/second_layer/leaky_relu/alpha:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
Т
!Generator/second_layer/leaky_reluMaximum%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
п
MGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      
б
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *ѓЕН*
dtype0*
_output_shapes
: 
б
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *ѓЕ=
Ч
UGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
seed2 
Ю
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
_output_shapes
: 
т
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/sub*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
д
GGenerator/third_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

х
,Generator/third_layer/fully_connected/kernel
VariableV2*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Щ
3Generator/third_layer/fully_connected/kernel/AssignAssign,Generator/third_layer/fully_connected/kernelGGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

з
1Generator/third_layer/fully_connected/kernel/readIdentity,Generator/third_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

Ъ
<Generator/third_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
з
*Generator/third_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:
Г
1Generator/third_layer/fully_connected/bias/AssignAssign*Generator/third_layer/fully_connected/bias<Generator/third_layer/fully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
Ь
/Generator/third_layer/fully_connected/bias/readIdentity*Generator/third_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:
х
,Generator/third_layer/fully_connected/MatMulMatMul!Generator/second_layer/leaky_relu1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
с
-Generator/third_layer/fully_connected/BiasAddBiasAdd,Generator/third_layer/fully_connected/MatMul/Generator/third_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
г
@Generator/third_layer/batch_normalization/gamma/Initializer/onesConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
с
/Generator/third_layer/batch_normalization/gamma
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma
Ц
6Generator/third_layer/batch_normalization/gamma/AssignAssign/Generator/third_layer/batch_normalization/gamma@Generator/third_layer/batch_normalization/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma
л
4Generator/third_layer/batch_normalization/gamma/readIdentity/Generator/third_layer/batch_normalization/gamma*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
в
@Generator/third_layer/batch_normalization/beta/Initializer/zerosConst*
_output_shapes	
:*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    *
dtype0
п
.Generator/third_layer/batch_normalization/beta
VariableV2*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
У
5Generator/third_layer/batch_normalization/beta/AssignAssign.Generator/third_layer/batch_normalization/beta@Generator/third_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
и
3Generator/third_layer/batch_normalization/beta/readIdentity.Generator/third_layer/batch_normalization/beta*
_output_shapes	
:*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
р
GGenerator/third_layer/batch_normalization/moving_mean/Initializer/zerosConst*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
э
5Generator/third_layer/batch_normalization/moving_mean
VariableV2*
shared_name *H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
п
<Generator/third_layer/batch_normalization/moving_mean/AssignAssign5Generator/third_layer/batch_normalization/moving_meanGGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean
э
:Generator/third_layer/batch_normalization/moving_mean/readIdentity5Generator/third_layer/batch_normalization/moving_mean*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
_output_shapes	
:*
T0
ч
JGenerator/third_layer/batch_normalization/moving_variance/Initializer/onesConst*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ѕ
9Generator/third_layer/batch_normalization/moving_variance
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance
ю
@Generator/third_layer/batch_normalization/moving_variance/AssignAssign9Generator/third_layer/batch_normalization/moving_varianceJGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(
љ
>Generator/third_layer/batch_normalization/moving_variance/readIdentity9Generator/third_layer/batch_normalization/moving_variance*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
_output_shapes	
:
~
9Generator/third_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
п
7Generator/third_layer/batch_normalization/batchnorm/addAdd>Generator/third_layer/batch_normalization/moving_variance/read9Generator/third_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:*
T0
Ё
9Generator/third_layer/batch_normalization/batchnorm/RsqrtRsqrt7Generator/third_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
е
7Generator/third_layer/batch_normalization/batchnorm/mulMul9Generator/third_layer/batch_normalization/batchnorm/Rsqrt4Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
л
9Generator/third_layer/batch_normalization/batchnorm/mul_1Mul-Generator/third_layer/fully_connected/BiasAdd7Generator/third_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:џџџџџџџџџ*
T0
л
9Generator/third_layer/batch_normalization/batchnorm/mul_2Mul:Generator/third_layer/batch_normalization/moving_mean/read7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
д
7Generator/third_layer/batch_normalization/batchnorm/subSub3Generator/third_layer/batch_normalization/beta/read9Generator/third_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:*
T0
ч
9Generator/third_layer/batch_normalization/batchnorm/add_1Add9Generator/third_layer/batch_normalization/batchnorm/mul_17Generator/third_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:џџџџџџџџџ*
T0
k
&Generator/third_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
С
$Generator/third_layer/leaky_relu/mulMul&Generator/third_layer/leaky_relu/alpha9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
П
 Generator/third_layer/leaky_reluMaximum$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0
н
LGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Я
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  Н*
dtype0*
_output_shapes
: 
Я
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
Ф
TGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
Ъ
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/subSubJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
_output_shapes
: 
о
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

а
FGenerator/last_layer/fully_connected/kernel/Initializer/random_uniformAddJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
у
+Generator/last_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:

Х
2Generator/last_layer/fully_connected/kernel/AssignAssign+Generator/last_layer/fully_connected/kernelFGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

д
0Generator/last_layer/fully_connected/kernel/readIdentity+Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
д
KGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0
Ф
AGenerator/last_layer/fully_connected/bias/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Щ
;Generator/last_layer/fully_connected/bias/Initializer/zerosFillKGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorAGenerator/last_layer/fully_connected/bias/Initializer/zeros/Const*
_output_shapes	
:*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0
е
)Generator/last_layer/fully_connected/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
Џ
0Generator/last_layer/fully_connected/bias/AssignAssign)Generator/last_layer/fully_connected/bias;Generator/last_layer/fully_connected/bias/Initializer/zeros*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Щ
.Generator/last_layer/fully_connected/bias/readIdentity)Generator/last_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:
т
+Generator/last_layer/fully_connected/MatMulMatMul Generator/third_layer/leaky_relu0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
о
,Generator/last_layer/fully_connected/BiasAddBiasAdd+Generator/last_layer/fully_connected/MatMul.Generator/last_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
н
OGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0*
_output_shapes
:
Э
EGenerator/last_layer/batch_normalization/gamma/Initializer/ones/ConstConst*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *  ?*
dtype0
к
?Generator/last_layer/batch_normalization/gamma/Initializer/onesFillOGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorEGenerator/last_layer/batch_normalization/gamma/Initializer/ones/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:
п
.Generator/last_layer/batch_normalization/gamma
VariableV2*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
Т
5Generator/last_layer/batch_normalization/gamma/AssignAssign.Generator/last_layer/batch_normalization/gamma?Generator/last_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
и
3Generator/last_layer/batch_normalization/gamma/readIdentity.Generator/last_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:
м
OGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:*
dtype0
Ь
EGenerator/last_layer/batch_normalization/beta/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
й
?Generator/last_layer/batch_normalization/beta/Initializer/zerosFillOGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorEGenerator/last_layer/batch_normalization/beta/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:
н
-Generator/last_layer/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:
П
4Generator/last_layer/batch_normalization/beta/AssignAssign-Generator/last_layer/batch_normalization/beta?Generator/last_layer/batch_normalization/beta/Initializer/zeros*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
е
2Generator/last_layer/batch_normalization/beta/readIdentity-Generator/last_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:
ъ
VGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB:*
dtype0*
_output_shapes
:
к
LGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/ConstConst*
_output_shapes
: *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB
 *    *
dtype0
ѕ
FGenerator/last_layer/batch_normalization/moving_mean/Initializer/zerosFillVGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*

index_type0*
_output_shapes	
:
ы
4Generator/last_layer/batch_normalization/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
	container 
л
;Generator/last_layer/batch_normalization/moving_mean/AssignAssign4Generator/last_layer/batch_normalization/moving_meanFGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:
ъ
9Generator/last_layer/batch_normalization/moving_mean/readIdentity4Generator/last_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
_output_shapes	
:
ё
YGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB:*
dtype0*
_output_shapes
:
с
OGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/ConstConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 

IGenerator/last_layer/batch_normalization/moving_variance/Initializer/onesFillYGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorOGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/Const*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*

index_type0*
_output_shapes	
:
ѓ
8Generator/last_layer/batch_normalization/moving_variance
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance
ъ
?Generator/last_layer/batch_normalization/moving_variance/AssignAssign8Generator/last_layer/batch_normalization/moving_varianceIGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:
і
=Generator/last_layer/batch_normalization/moving_variance/readIdentity8Generator/last_layer/batch_normalization/moving_variance*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
_output_shapes	
:
}
8Generator/last_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
м
6Generator/last_layer/batch_normalization/batchnorm/addAdd=Generator/last_layer/batch_normalization/moving_variance/read8Generator/last_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:*
T0

8Generator/last_layer/batch_normalization/batchnorm/RsqrtRsqrt6Generator/last_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
в
6Generator/last_layer/batch_normalization/batchnorm/mulMul8Generator/last_layer/batch_normalization/batchnorm/Rsqrt3Generator/last_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
и
8Generator/last_layer/batch_normalization/batchnorm/mul_1Mul,Generator/last_layer/fully_connected/BiasAdd6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
и
8Generator/last_layer/batch_normalization/batchnorm/mul_2Mul9Generator/last_layer/batch_normalization/moving_mean/read6Generator/last_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:*
T0
б
6Generator/last_layer/batch_normalization/batchnorm/subSub2Generator/last_layer/batch_normalization/beta/read8Generator/last_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ф
8Generator/last_layer/batch_normalization/batchnorm/add_1Add8Generator/last_layer/batch_normalization/batchnorm/mul_16Generator/last_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:џџџџџџџџџ
j
%Generator/last_layer/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
О
#Generator/last_layer/leaky_relu/mulMul%Generator/last_layer/leaky_relu/alpha8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
М
Generator/last_layer/leaky_reluMaximum#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ
Н
<Generator/fake_image/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
Џ
:Generator/fake_image/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zѕkН*
dtype0*
_output_shapes
: 
Џ
:Generator/fake_image/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zѕk=*
dtype0*
_output_shapes
: 

DGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniformRandomUniform<Generator/fake_image/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*.
_class$
" loc:@Generator/fake_image/kernel*
seed2 

:Generator/fake_image/kernel/Initializer/random_uniform/subSub:Generator/fake_image/kernel/Initializer/random_uniform/max:Generator/fake_image/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
_output_shapes
: 

:Generator/fake_image/kernel/Initializer/random_uniform/mulMulDGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniform:Generator/fake_image/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:


6Generator/fake_image/kernel/Initializer/random_uniformAdd:Generator/fake_image/kernel/Initializer/random_uniform/mul:Generator/fake_image/kernel/Initializer/random_uniform/min*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
*
T0
У
Generator/fake_image/kernel
VariableV2*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:


"Generator/fake_image/kernel/AssignAssignGenerator/fake_image/kernel6Generator/fake_image/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel
Є
 Generator/fake_image/kernel/readIdentityGenerator/fake_image/kernel* 
_output_shapes
:
*
T0*.
_class$
" loc:@Generator/fake_image/kernel
Ј
+Generator/fake_image/bias/Initializer/zerosConst*
_output_shapes	
:*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0
Е
Generator/fake_image/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@Generator/fake_image/bias
я
 Generator/fake_image/bias/AssignAssignGenerator/fake_image/bias+Generator/fake_image/bias/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:

Generator/fake_image/bias/readIdentityGenerator/fake_image/bias*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
С
Generator/fake_image/MatMulMatMulGenerator/last_layer/leaky_relu Generator/fake_image/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ў
Generator/fake_image/BiasAddBiasAddGenerator/fake_image/MatMulGenerator/fake_image/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ
r
Generator/fake_image/TanhTanhGenerator/fake_image/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
z
Discriminator/real_inPlaceholder*
dtype0*(
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
ч
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
й
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HYН*
dtype0*
_output_shapes
: 
й
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY=
г
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*

seed *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:

о
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
ђ
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
ф
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
э
0Discriminator/first_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:

й
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

у
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:

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
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
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
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:
с
0Discriminator/first_layer/fully_connected/MatMulMatMulDiscriminator/real_in5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
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
ZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformRDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
т
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
_output_shapes
: *
T0
і
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
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
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container 
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
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
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
%Discriminator/second_layer/leaky_reluMaximum)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0
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
8Discriminator/prob/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Iv>

BDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Discriminator/prob/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
seed2 
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
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	
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
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container 
ц
Discriminator/prob/bias/AssignAssignDiscriminator/prob/bias)Discriminator/prob/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(

Discriminator/prob/bias/readIdentityDiscriminator/prob/bias*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
Т
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
Ї
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:џџџџџџџџџ
ч
2Discriminator/first_layer_1/fully_connected/MatMulMatMulGenerator/fake_image/Tanh5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
ё
3Discriminator/first_layer_1/fully_connected/BiasAddBiasAdd2Discriminator/first_layer_1/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:џџџџџџџџџ*
T0
q
,Discriminator/first_layer_1/leaky_relu/alphaConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: 
Ч
*Discriminator/first_layer_1/leaky_relu/mulMul,Discriminator/first_layer_1/leaky_relu/alpha3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
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
Discriminator/prob_1/BiasAddBiasAddDiscriminator/prob_1/MatMulDiscriminator/prob/bias/read*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
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
 *  ?
w
	ones_likeFillones_like/Shapeones_like/Const*

index_type0*'
_output_shapes
:џџџџџџџџџ*
T0
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
logistic_loss/SelectSelectlogistic_loss/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
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
logistic_loss/mulMulDiscriminator/prob/BiasAdd	ones_like*'
_output_shapes
:џџџџџџџџџ*
T0
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*'
_output_shapes
:џџџџџџџџџ*
T0
b
logistic_loss/ExpExplogistic_loss/Select_1*'
_output_shapes
:џџџџџџџџџ*
T0
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
`
MeanMeanlogistic_lossConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
g

zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
w
logistic_loss_1/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ

logistic_loss_1/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
Њ
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
j
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
Ѕ
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
v
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAdd
zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
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
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*
T0*'
_output_shapes
:џџџџџџџџџ
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*
T0*'
_output_shapes
:џџџџџџџџџ
X
Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
f
Mean_1Meanlogistic_loss_1Const_1*
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
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0

logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:џџџџџџџџџ*
T0
j
logistic_loss_2/NegNegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
Ѕ
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
w
logistic_loss_2/mulMulDiscriminator/prob_1/BiasAddones_like_1*
T0*'
_output_shapes
:џџџџџџџџџ
y
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*
T0*'
_output_shapes
:џџџџџџџџџ
f
logistic_loss_2/ExpExplogistic_loss_2/Select_1*
T0*'
_output_shapes
:џџџџџџџџџ
e
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*'
_output_shapes
:џџџџџџџџџ*
T0
t
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*
T0*'
_output_shapes
:џџџџџџџџџ
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
 *  ?*
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
Б
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Fill$^gradients/add_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
_output_shapes
: *
T0
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
gradients/Mean_grad/ReshapeReshape+gradients/add_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
f
gradients/Mean_grad/ShapeShapelogistic_loss*
_output_shapes
:*
T0*
out_type0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
h
gradients/Mean_grad/Shape_1Shapelogistic_loss*
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

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

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

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
t
#gradients/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
Г
gradients/Mean_1_grad/ReshapeReshape-gradients/add_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
j
gradients/Mean_1_grad/ShapeShapelogistic_loss_1*
T0*
out_type0*
_output_shapes
:
Ђ
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*
T0*'
_output_shapes
:џџџџџџџџџ*

Tmultiples0
l
gradients/Mean_1_grad/Shape_1Shapelogistic_loss_1*
_output_shapes
:*
T0*
out_type0
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
s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
_output_shapes
:*
T0*
out_type0
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:
в
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
И
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Е
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
М
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1

5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
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
&gradients/logistic_loss_1_grad/Shape_1Shapelogistic_loss_1/Log1p*
out_type0*
_output_shapes
:*
T0
и
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
О
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Л
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Т
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
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
7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1
z
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
_output_shapes
:*
T0*
out_type0
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
out_type0*
_output_shapes
:*
T0
о
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
к
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
С
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
о
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
Х
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1

9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ї
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0*
_output_shapes
: 

&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*'
_output_shapes
:џџџџџџџџџ*
T0
Ч
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ
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
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1
Ћ
*gradients/logistic_loss_1/Log1p_grad/add/xConst:^gradients/logistic_loss_1_grad/tuple/control_dependency_1*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ђ
(gradients/logistic_loss_1/Log1p_grad/addAdd*gradients/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*'
_output_shapes
:џџџџџџџџџ*
T0

/gradients/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*
T0*'
_output_shapes
:џџџџџџџџџ
Э
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ

.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
э
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
я
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1

<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Ђ
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:џџџџџџџџџ

&gradients/logistic_loss/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
T0*
out_type0*
_output_shapes
:
q
(gradients/logistic_loss/mul_grad/Shape_1Shape	ones_like*
_output_shapes
:*
T0*
out_type0
о
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*
T0*'
_output_shapes
:џџџџџџџџџ
Щ
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
С
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
И
&gradients/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*'
_output_shapes
:џџџџџџџџџ*
T0
Я
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ч
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1

9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

0gradients/logistic_loss_1/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
ѕ
,gradients/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual;gradients/logistic_loss_1/sub_grad/tuple/control_dependency0gradients/logistic_loss_1/Select_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
ї
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
Є
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Њ
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_grad/Select_1*'
_output_shapes
:џџџџџџџџџ

(gradients/logistic_loss_1/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
out_type0*
_output_shapes
:*
T0
t
*gradients/logistic_loss_1/mul_grad/Shape_1Shape
zeros_like*
T0*
out_type0*
_output_shapes
:
ф
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Њ
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1
zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
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

0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*'
_output_shapes
:џџџџџџџџџ*
T0
м
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
о
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ

6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
Є
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select*'
_output_shapes
:џџџџџџџџџ
Њ
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1*'
_output_shapes
:џџџџџџџџџ*
T0

2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*
T0*'
_output_shapes
:џџџџџџџџџ
ф
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
ц
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*'
_output_shapes
:џџџџџџџџџ*
T0
Є
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
Ќ
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select
В
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1*'
_output_shapes
:џџџџџџџџџ

$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ
ё
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*'
_output_shapes
:џџџџџџџџџ*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N

5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:

:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN6^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad

Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
Г
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
§
gradients/AddN_1AddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N*'
_output_shapes
:џџџџџџџџџ

7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
data_formatNHWC*
_output_shapes
:*
T0

<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_18^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad

Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select
Л
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
і
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
і
1gradients/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
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
Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	

gradients/AddN_2AddNDgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
Ѓ
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ў
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Н
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2ShapeAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
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
;gradients/Discriminator/second_layer/leaky_relu_grad/SelectSelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency:gradients/Discriminator/second_layer/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
М
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

8gradients/Discriminator/second_layer/leaky_relu_grad/SumSum;gradients/Discriminator/second_layer/leaky_relu_grad/SelectJgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ў
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1Lgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Э
Egradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp=^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape?^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
у
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape
щ
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Ї
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
В
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
С
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    

<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosFill>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
щ
Cgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
 
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Т
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
Ф
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
г
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
ы
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape
ё
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

gradients/AddN_3AddNCgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
N*
_output_shapes
:	

>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ј
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
є
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
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
Qgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeJ^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*S
_classI
GEloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
љ
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
Dgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1Reshape@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
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
Э
gradients/AddN_4AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Ћ
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:
Н
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
г
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ*
T0

\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
г
gradients/AddN_5AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0
­
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
data_formatNHWC*
_output_shapes	
:*
T0
С
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
й
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
О
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
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
[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
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
ч
gradients/AddN_6AddN\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
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
9gradients/Discriminator/first_layer/leaky_relu_grad/zerosFill;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_2?gradients/Discriminator/first_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
р
@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
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
=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ъ
Dgradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp<^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape>^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
п
Lgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeE^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*N
_classD
B@loc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ*
T0
х
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Ѕ
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
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
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
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
9gradients/Discriminator/first_layer_1/leaky_relu_grad/SumSum<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectKgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
ц
gradients/AddN_7AddN[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:

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
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
і
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:џџџџџџџџџ*
T0

;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Ogradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Agradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
ж
Hgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp@^gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeB^gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
н
Pgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeI^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape
ѕ
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
Ъ
gradients/AddN_8AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
Њ
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:
Л
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
а
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
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
а
gradients/AddN_9AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0
Ќ
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:
П
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ж
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
Л
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0

Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/real_inYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ь
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1

Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
П
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
ђ
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1

Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul

\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1S^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

х
gradients/AddN_10AddN[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
ф
gradients/AddN_11AddNZgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:

Ё
beta1_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *fff?*
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0

beta2_power/readIdentitybeta2_power*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
э
WDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
з
MDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
љ
GDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillWDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorMDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ђ
5Discriminator/first_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:

п
<Discriminator/first_layer/fully_connected/kernel/Adam/AssignAssign5Discriminator/first_layer/fully_connected/kernel/AdamGDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(
э
:Discriminator/first_layer/fully_connected/kernel/Adam/readIdentity5Discriminator/first_layer/fully_connected/kernel/Adam*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:

я
YDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     
й
ODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
џ
IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillYDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

є
7Discriminator/first_layer/fully_connected/kernel/Adam_1
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container 
х
>Discriminator/first_layer/fully_connected/kernel/Adam_1/AssignAssign7Discriminator/first_layer/fully_connected/kernel/Adam_1IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

ё
<Discriminator/first_layer/fully_connected/kernel/Adam_1/readIdentity7Discriminator/first_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
з
EDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ф
3Discriminator/first_layer/fully_connected/bias/Adam
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
в
:Discriminator/first_layer/fully_connected/bias/Adam/AssignAssign3Discriminator/first_layer/fully_connected/bias/AdamEDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
т
8Discriminator/first_layer/fully_connected/bias/Adam/readIdentity3Discriminator/first_layer/fully_connected/bias/Adam*
_output_shapes	
:*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
й
GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    
ц
5Discriminator/first_layer/fully_connected/bias/Adam_1
VariableV2*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
и
<Discriminator/first_layer/fully_connected/bias/Adam_1/AssignAssign5Discriminator/first_layer/fully_connected/bias/Adam_1GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(
ц
:Discriminator/first_layer/fully_connected/bias/Adam_1/readIdentity5Discriminator/first_layer/fully_connected/bias/Adam_1*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:*
T0
я
XDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
й
NDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
§
HDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillXDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorNDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

є
6Discriminator/second_layer/fully_connected/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container 
у
=Discriminator/second_layer/fully_connected/kernel/Adam/AssignAssign6Discriminator/second_layer/fully_connected/kernel/AdamHDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
№
;Discriminator/second_layer/fully_connected/kernel/Adam/readIdentity6Discriminator/second_layer/fully_connected/kernel/Adam*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

ё
ZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      
л
PDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0

JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorPDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

і
8Discriminator/second_layer/fully_connected/kernel/Adam_1
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
щ
?Discriminator/second_layer/fully_connected/kernel/Adam_1/AssignAssign8Discriminator/second_layer/fully_connected/kernel/Adam_1JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

є
=Discriminator/second_layer/fully_connected/kernel/Adam_1/readIdentity8Discriminator/second_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
й
FDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ц
4Discriminator/second_layer/fully_connected/bias/Adam
VariableV2*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ж
;Discriminator/second_layer/fully_connected/bias/Adam/AssignAssign4Discriminator/second_layer/fully_connected/bias/AdamFDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(
х
9Discriminator/second_layer/fully_connected/bias/Adam/readIdentity4Discriminator/second_layer/fully_connected/bias/Adam*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
л
HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ш
6Discriminator/second_layer/fully_connected/bias/Adam_1
VariableV2*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
м
=Discriminator/second_layer/fully_connected/bias/Adam_1/AssignAssign6Discriminator/second_layer/fully_connected/bias/Adam_1HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
щ
;Discriminator/second_layer/fully_connected/bias/Adam_1/readIdentity6Discriminator/second_layer/fully_connected/bias/Adam_1*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
Е
0Discriminator/prob/kernel/Adam/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Т
Discriminator/prob/kernel/Adam
VariableV2*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel

%Discriminator/prob/kernel/Adam/AssignAssignDiscriminator/prob/kernel/Adam0Discriminator/prob/kernel/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	
Ї
#Discriminator/prob/kernel/Adam/readIdentityDiscriminator/prob/kernel/Adam*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	
З
2Discriminator/prob/kernel/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Ф
 Discriminator/prob/kernel/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	

'Discriminator/prob/kernel/Adam_1/AssignAssign Discriminator/prob/kernel/Adam_12Discriminator/prob/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	
Ћ
%Discriminator/prob/kernel/Adam_1/readIdentity Discriminator/prob/kernel/Adam_1*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	
Ї
.Discriminator/prob/bias/Adam/Initializer/zerosConst*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0
Д
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
ѕ
#Discriminator/prob/bias/Adam/AssignAssignDiscriminator/prob/bias/Adam.Discriminator/prob/bias/Adam/Initializer/zeros*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:*
use_locking(

!Discriminator/prob/bias/Adam/readIdentityDiscriminator/prob/bias/Adam**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:*
T0
Љ
0Discriminator/prob/bias/Adam_1/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
Ж
Discriminator/prob/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container 
ћ
%Discriminator/prob/bias/Adam_1/AssignAssignDiscriminator/prob/bias/Adam_10Discriminator/prob/bias/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
 
#Discriminator/prob/bias/Adam_1/readIdentityDiscriminator/prob/bias/Adam_1*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *ЗQ9
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
 *wО?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
§
FAdam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam0Discriminator/first_layer/fully_connected/kernel5Discriminator/first_layer/fully_connected/kernel/Adam7Discriminator/first_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_11*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
ю
DAdam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam.Discriminator/first_layer/fully_connected/bias3Discriminator/first_layer/fully_connected/bias/Adam5Discriminator/first_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_10*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 

GAdam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernel6Discriminator/second_layer/fully_connected/kernel/Adam8Discriminator/second_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7* 
_output_shapes
:
*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( 
ђ
EAdam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/bias4Discriminator/second_layer/fully_connected/bias/Adam6Discriminator/second_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 

/Adam/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernelDiscriminator/prob/kernel/Adam Discriminator/prob/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
use_nesterov( *
_output_shapes
:	
љ
-Adam/update_Discriminator/prob/bias/ApplyAdam	ApplyAdamDiscriminator/prob/biasDiscriminator/prob/bias/AdamDiscriminator/prob/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0**
_class 
loc:@Discriminator/prob/bias

Adam/mulMulbeta1_power/read
Adam/beta1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
Й
Adam/AssignAssignbeta1_powerAdam/mul*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0


Adam/mul_1Mulbeta2_power/read
Adam/beta2E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
Н
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
Ў
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
 *  ?*
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

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
Ј
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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
Ђ
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
І
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

gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
_output_shapes
: *
T0

gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*'
_output_shapes
:џџџџџџџџџ*
T0
y
&gradients_1/logistic_loss_2_grad/ShapeShapelogistic_loss_2/sub*
T0*
out_type0*
_output_shapes
:
}
(gradients_1/logistic_loss_2_grad/Shape_1Shapelogistic_loss_2/Log1p*
_output_shapes
:*
T0*
out_type0
о
6gradients_1/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_2_grad/Shape(gradients_1/logistic_loss_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ф
$gradients_1/logistic_loss_2_grad/SumSumgradients_1/Mean_2_grad/truediv6gradients_1/logistic_loss_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
С
(gradients_1/logistic_loss_2_grad/ReshapeReshape$gradients_1/logistic_loss_2_grad/Sum&gradients_1/logistic_loss_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ш
&gradients_1/logistic_loss_2_grad/Sum_1Sumgradients_1/Mean_2_grad/truediv8gradients_1/logistic_loss_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ч
*gradients_1/logistic_loss_2_grad/Reshape_1Reshape&gradients_1/logistic_loss_2_grad/Sum_1(gradients_1/logistic_loss_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

1gradients_1/logistic_loss_2_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_2_grad/Reshape+^gradients_1/logistic_loss_2_grad/Reshape_1

9gradients_1/logistic_loss_2_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_2_grad/Reshape2^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџ

;gradients_1/logistic_loss_2_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_2_grad/Reshape_12^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss_2_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ

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
ъ
:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/sub_grad/Shape,gradients_1/logistic_loss_2/sub_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ц
(gradients_1/logistic_loss_2/sub_grad/SumSum9gradients_1/logistic_loss_2_grad/tuple/control_dependency:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Э
,gradients_1/logistic_loss_2/sub_grad/ReshapeReshape(gradients_1/logistic_loss_2/sub_grad/Sum*gradients_1/logistic_loss_2/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
ъ
*gradients_1/logistic_loss_2/sub_grad/Sum_1Sum9gradients_1/logistic_loss_2_grad/tuple/control_dependency<gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
~
(gradients_1/logistic_loss_2/sub_grad/NegNeg*gradients_1/logistic_loss_2/sub_grad/Sum_1*
T0*
_output_shapes
:
б
.gradients_1/logistic_loss_2/sub_grad/Reshape_1Reshape(gradients_1/logistic_loss_2/sub_grad/Neg,gradients_1/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5gradients_1/logistic_loss_2/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/sub_grad/Reshape/^gradients_1/logistic_loss_2/sub_grad/Reshape_1
Ђ
=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/sub_grad/Reshape6^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/sub_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ј
?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/sub_grad/Reshape_16^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/sub_grad/Reshape_1
Џ
,gradients_1/logistic_loss_2/Log1p_grad/add/xConst<^gradients_1/logistic_loss_2_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  ?
І
*gradients_1/logistic_loss_2/Log1p_grad/addAdd,gradients_1/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

1gradients_1/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_2/Log1p_grad/add*
T0*'
_output_shapes
:џџџџџџџџџ
г
*gradients_1/logistic_loss_2/Log1p_grad/mulMul;gradients_1/logistic_loss_2_grad/tuple/control_dependency_11gradients_1/logistic_loss_2/Log1p_grad/Reciprocal*'
_output_shapes
:џџџџџџџџџ*
T0

2gradients_1/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
ћ
.gradients_1/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_2/Select_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
§
0gradients_1/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients_1/logistic_loss_2/Select_grad/zeros_like=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency*'
_output_shapes
:џџџџџџџџџ*
T0
Є
8gradients_1/logistic_loss_2/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_2/Select_grad/Select1^gradients_1/logistic_loss_2/Select_grad/Select_1
Ќ
@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_2/Select_grad/Select9^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:џџџџџџџџџ
В
Bgradients_1/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_2/Select_grad/Select_19^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_grad/Select_1

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
ъ
:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/mul_grad/Shape,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Џ
(gradients_1/logistic_loss_2/mul_grad/MulMul?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1ones_like_1*'
_output_shapes
:џџџџџџџџџ*
T0
е
(gradients_1/logistic_loss_2/mul_grad/SumSum(gradients_1/logistic_loss_2/mul_grad/Mul:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Э
,gradients_1/logistic_loss_2/mul_grad/ReshapeReshape(gradients_1/logistic_loss_2/mul_grad/Sum*gradients_1/logistic_loss_2/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Т
*gradients_1/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:џџџџџџџџџ
л
*gradients_1/logistic_loss_2/mul_grad/Sum_1Sum*gradients_1/logistic_loss_2/mul_grad/Mul_1<gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
г
.gradients_1/logistic_loss_2/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_2/mul_grad/Sum_1,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ

5gradients_1/logistic_loss_2/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/mul_grad/Reshape/^gradients_1/logistic_loss_2/mul_grad/Reshape_1
Ђ
=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/mul_grad/Reshape6^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
Ј
?gradients_1/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/mul_grad/Reshape_16^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ
Ђ
(gradients_1/logistic_loss_2/Exp_grad/mulMul*gradients_1/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*
T0*'
_output_shapes
:џџџџџџџџџ

4gradients_1/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*
T0*'
_output_shapes
:џџџџџџџџџ
ъ
0gradients_1/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual(gradients_1/logistic_loss_2/Exp_grad/mul4gradients_1/logistic_loss_2/Select_1_grad/zeros_like*
T0*'
_output_shapes
:џџџџџџџџџ
ь
2gradients_1/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual4gradients_1/logistic_loss_2/Select_1_grad/zeros_like(gradients_1/logistic_loss_2/Exp_grad/mul*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
:gradients_1/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_2/Select_1_grad/Select3^gradients_1/logistic_loss_2/Select_1_grad/Select_1
Д
Bgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_2/Select_1_grad/Select;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_1_grad/Select*'
_output_shapes
:џџџџџџџџџ
К
Dgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_2/Select_1_grad/Select_1;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/logistic_loss_2/Select_1_grad/Select_1*'
_output_shapes
:џџџџџџџџџ
Ѕ
(gradients_1/logistic_loss_2/Neg_grad/NegNegBgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:џџџџџџџџџ

gradients_1/AddNAddN@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_2/Neg_grad/Neg*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*
N*'
_output_shapes
:џџџџџџџџџ

9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
data_formatNHWC*
_output_shapes
:*
T0

>gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN:^gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad

Fgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select
У
Hgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
ў
3gradients_1/Discriminator/prob_1/MatMul_grad/MatMulMatMulFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
Г
=gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul6^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1
С
Egradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Discriminator/prob_1/MatMul_grad/MatMul>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
О
Ggradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*H
_class>
<:loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1
Љ
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Д
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Х
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosFill@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
ы
Egradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
І
Ngradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ъ
?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
Ь
Agradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SumSum?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectNgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0

>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1SumAgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1Pgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Bgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
й
Igradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOpA^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeC^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
ѓ
Qgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeJ^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
љ
Sgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityBgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1J^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1

Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
И
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
В
Rgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulRgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
ў
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
Ѓ
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Tgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Fgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
х
Mgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpE^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeG^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
ё
Ugradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeN^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*W
_classM
KIloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0

Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityFgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1N^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
л
gradients_1/AddN_1AddNSgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Б
Qgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
data_formatNHWC*
_output_shapes	
:*
T0
Ч
Vgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1R^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
с
^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1W^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Є
`gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityQgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*d
_classZ
XVloc:@gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
Ц
Kgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
А
Mgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ћ
Ugradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpL^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulN^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
Ё
]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityKgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulV^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

_gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityMgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1V^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*`
_classV
TRloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
Ї
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
В
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
м
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zerosFill?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
ш
Dgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ
Ѓ
Mgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
п
>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
с
@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SumSum>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectMgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1Ogradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Agradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
ж
Hgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp@^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeB^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
я
Pgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeI^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ѕ
Rgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityAgradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1I^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ

Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ж
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Џ
Qgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulQgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ћ
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
 
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Sgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Egradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
т
Lgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpD^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeF^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
э
Tgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeM^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*V
_classL
JHloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0

Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityEgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1M^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*X
_classN
LJloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
и
gradients_1/AddN_2AddNRgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
А
Pgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:
Х
Ugradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2Q^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
о
]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2V^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
 
_gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradV^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
У
Jgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
Ё
Lgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ј
Tgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpK^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulM^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1

\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityJgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulU^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityLgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1U^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

ы
3gradients_1/Generator/fake_image/Tanh_grad/TanhGradTanhGradGenerator/fake_image/Tanh\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
К
9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:
И
>gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad4^gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
У
Fgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*(
_output_shapes
:џџџџџџџџџ
Ф
Hgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0

3gradients_1/Generator/fake_image/MatMul_grad/MatMulMatMulFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency Generator/fake_image/kernel/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
љ
5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1MatMulGenerator/last_layer/leaky_reluFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
Г
=gradients_1/Generator/fake_image/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Generator/fake_image/MatMul_grad/MatMul6^gradients_1/Generator/fake_image/MatMul_grad/MatMul_1
С
Egradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/MatMul_grad/MatMul>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
П
Ggradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul_1* 
_output_shapes
:


6gradients_1/Generator/last_layer/leaky_relu_grad/ShapeShape#Generator/last_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
А
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Н
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2ShapeEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
ћ
6gradients_1/Generator/last_layer/leaky_relu_grad/zerosFill8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
п
=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

Fgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Generator/last_layer/leaky_relu_grad/Shape8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
В
7gradients_1/Generator/last_layer/leaky_relu_grad/SelectSelect=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency6gradients_1/Generator/last_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
Д
9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Select=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqual6gradients_1/Generator/last_layer/leaky_relu_grad/zerosEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
ќ
4gradients_1/Generator/last_layer/leaky_relu_grad/SumSum7gradients_1/Generator/last_layer/leaky_relu_grad/SelectFgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ђ
8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Generator/last_layer/leaky_relu_grad/Sum6gradients_1/Generator/last_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Hgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ј
:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_18gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
С
Agradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape;^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1
г
Igradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeB^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
й
Kgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1B^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
}
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Д
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

Jgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ї
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulMulIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

8gradients_1/Generator/last_layer/leaky_relu/mul_grad/SumSum8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulJgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ь
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ц
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Mul%Generator/last_layer/leaky_relu/alphaIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Э
Egradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
б
Mgradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*O
_classE
CAloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape
щ
Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
У
gradients_1/AddN_3AddNKgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:џџџџџџџџџ*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*
N
Ч
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Generator/last_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:

Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
й
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_3_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Н
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_3agradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ж
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
З
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ*
T0
А
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Л
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Generator/last_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
й
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѓ
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
Ф
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Н
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Generator/last_layer/fully_connected/BiasAddbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Ъ
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
З
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
А
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
о
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:*
T0

Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg
Л
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:

bgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:*
T0
љ
Igradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Ngradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad
А
Vgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ

Xgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*\
_classR
PNloc:@gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad

Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Generator/last_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Generator/last_layer/batch_normalization/moving_mean/read*
_output_shapes	
:*
T0

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Ђ
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul
Ј
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
А
Cgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

Egradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/third_layer/leaky_reluVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
у
Mgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1

Ugradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
џ
Wgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*X
_classN
LJloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1
§
gradients_1/AddN_4AddNdgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
С
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_43Generator/last_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
Ш
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_48Generator/last_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:*
T0
ў
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1

`gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:
 
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:*
T0

7gradients_1/Generator/third_layer/leaky_relu_grad/ShapeShape$Generator/third_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
В
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Ю
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0

=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ў
7gradients_1/Generator/third_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:џџџџџџџџџ*
T0
т
>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

Ggradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/third_layer/leaky_relu_grad/Shape9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Х
8gradients_1/Generator/third_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/third_layer/leaky_relu_grad/zeros*(
_output_shapes
:џџџџџџџџџ*
T0
Ч
:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/third_layer/leaky_relu_grad/zerosUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
џ
5gradients_1/Generator/third_layer/leaky_relu_grad/SumSum8gradients_1/Generator/third_layer/leaky_relu_grad/SelectGgradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ѕ
9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/third_layer/leaky_relu_grad/Sum7gradients_1/Generator/third_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Igradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ћ
;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_19gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ф
Bgradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
з
Jgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
н
Lgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
~
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ж
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

Kgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
њ
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0

9gradients_1/Generator/third_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
я
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
щ
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/third_layer/leaky_relu/alphaJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0

;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
а
Fgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
е
Ngradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
э
Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*R
_classH
FDloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ*
T0
Ц
gradients_1/AddN_5AddNLgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Щ
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape9Generator/third_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:

Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
м
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_5`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Р
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_5bgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Й
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
Л
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape
Д
egradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Н
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape-Generator/third_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
м
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
І
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
Ч
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumSumNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Р
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul-Generator/third_layer/fully_connected/BiasAddcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ
Э
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1bgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Й
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
Л
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape
Д
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:*
T0
р
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegNegegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpf^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1M^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg
П
agradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
 
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:
ћ
Jgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradcgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Ogradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpd^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyK^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Д
Wgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitycgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ

Ygradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_17Generator/third_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:*
T0

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1:Generator/third_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulQ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
І
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
Ќ
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*c
_classY
WUloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:*
T0
Г
Dgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(

Fgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1MatMul!Generator/second_layer/leaky_reluWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
ц
Ngradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1

Vgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

Xgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:


gradients_1/AddN_6AddNegradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
У
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_64Generator/third_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
Ъ
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_69Generator/third_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:

Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpM^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1

agradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:*
T0
Є
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:

8gradients_1/Generator/second_layer/leaky_relu_grad/ShapeShape%Generator/second_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Д
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
а
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2ShapeVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

8gradients_1/Generator/second_layer/leaky_relu_grad/zerosFill:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:џџџџџџџџџ
х
?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:џџџџџџџџџ

Hgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/Generator/second_layer/leaky_relu_grad/Shape:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Щ
9gradients_1/Generator/second_layer/leaky_relu_grad/SelectSelect?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency8gradients_1/Generator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
Ы
;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Select?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqual8gradients_1/Generator/second_layer/leaky_relu_grad/zerosVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

6gradients_1/Generator/second_layer/leaky_relu_grad/SumSum9gradients_1/Generator/second_layer/leaky_relu_grad/SelectHgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ј
:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeReshape6gradients_1/Generator/second_layer/leaky_relu_grad/Sum8gradients_1/Generator/second_layer/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0

8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1Sum;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Jgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ў
<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1Reshape8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ч
Cgradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_depsNoOp;^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape=^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1
л
Kgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeD^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
с
Mgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1D^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
И
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
 
Lgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
§
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulMulKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:џџџџџџџџџ*
T0

:gradients_1/Generator/second_layer/leaky_relu/mul_grad/SumSum:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulLgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ђ
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeReshape:gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ь
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Mul'Generator/second_layer/leaky_relu/alphaKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Ngradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
г
Ggradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp?^gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeA^gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
й
Ogradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeH^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
ё
Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1H^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Щ
gradients_1/AddN_7AddNMgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ*
T0
Ы
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape:Generator/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:

Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
п
agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_7agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
У
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ

Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_7cgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
М
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
П
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape
И
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
П
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape.Generator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:

Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
п
agradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Љ
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:џџџџџџџџџ
Ъ
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mulagradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
У
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ё
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul.Generator/second_layer/fully_connected/BiasAdddgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
а
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1cgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
М
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
П
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
И
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
т
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/NegNegfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:*
T0

Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpg^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1N^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
У
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
Є
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
§
Kgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGraddgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Pgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpe^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyL^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
И
Xgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitydgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:џџџџџџџџџ

Zgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_18Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
Ё
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Muldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1;Generator/second_layer/batch_normalization/moving_mean/read*
_output_shapes	
:*
T0

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulR^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Њ
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
А
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*d
_classZ
XVloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:*
T0
Ж
Egradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulMatMulXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency2Generator/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 

Ggradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/first_layer/leaky_reluXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
щ
Ogradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpF^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulH^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1

Wgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityEgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulP^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

Ygradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityGgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1P^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*Z
_classP
NLloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1

gradients_1/AddN_8AddNfgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
Х
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_85Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
Ь
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_8:Generator/second_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:*
T0

Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpN^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
Ђ
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul
Ј
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:

7gradients_1/Generator/first_layer/leaky_relu_grad/ShapeShape$Generator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
І
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
а
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2ShapeWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ў
7gradients_1/Generator/first_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:џџџџџџџџџ*
T0*

index_type0
ж
>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

Ggradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/first_layer/leaky_relu_grad/Shape9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ч
8gradients_1/Generator/first_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:џџџџџџџџџ
Щ
:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/first_layer/leaky_relu_grad/zerosWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:џџџџџџџџџ*
T0
џ
5gradients_1/Generator/first_layer/leaky_relu_grad/SumSum8gradients_1/Generator/first_layer/leaky_relu_grad/SelectGgradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ѕ
9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/first_layer/leaky_relu_grad/Sum7gradients_1/Generator/first_layer/leaky_relu_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Igradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ћ
;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_19gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ф
Bgradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
з
Jgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
н
Lgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
~
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Њ
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Kgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ю
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:џџџџџџџџџ

9gradients_1/Generator/first_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
я
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
щ
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/first_layer/leaky_relu/alphaJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:џџџџџџџџџ

;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
а
Fgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
е
Ngradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*P
_classF
DBloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape
э
Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:џџџџџџџџџ
Ц
gradients_1/AddN_9AddNLgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:џџџџџџџџџ
Њ
Jgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:
Й
Ogradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_9K^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ь
Wgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_9P^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1

Ygradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
В
Dgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/first_layer/fully_connected/kernel/read*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
transpose_b(*
T0

Fgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise_inWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d*
transpose_a(*
transpose_b( 
ц
Ngradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1

Vgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџd*
T0*W
_classM
KIloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul

Xgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d

beta1_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 

beta1_power_1
VariableV2*
_output_shapes
: *
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape: *
dtype0
Т
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

beta2_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *wО?*
dtype0*
_output_shapes
: 

beta2_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape: 
Т
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
х
SGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
Я
IGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
ш
CGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d
ш
1Generator/first_layer/fully_connected/kernel/Adam
VariableV2*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
Ю
8Generator/first_layer/fully_connected/kernel/Adam/AssignAssign1Generator/first_layer/fully_connected/kernel/AdamCGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d
р
6Generator/first_layer/fully_connected/kernel/Adam/readIdentity1Generator/first_layer/fully_connected/kernel/Adam*
_output_shapes
:	d*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
ч
UGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
б
KGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ю
EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d
ъ
3Generator/first_layer/fully_connected/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d
д
:Generator/first_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/first_layer/fully_connected/kernel/Adam_1EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0
ф
8Generator/first_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/first_layer/fully_connected/kernel/Adam_1*
_output_shapes
:	d*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
Я
AGenerator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
м
/Generator/first_layer/fully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:
Т
6Generator/first_layer/fully_connected/bias/Adam/AssignAssign/Generator/first_layer/fully_connected/bias/AdamAGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ж
4Generator/first_layer/fully_connected/bias/Adam/readIdentity/Generator/first_layer/fully_connected/bias/Adam*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:*
T0
б
CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
о
1Generator/first_layer/fully_connected/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
Ш
8Generator/first_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/first_layer/fully_connected/bias/Adam_1CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
к
6Generator/first_layer/fully_connected/bias/Adam_1/readIdentity1Generator/first_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:
ч
TGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
б
JGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
э
DGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillTGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorJGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ь
2Generator/second_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:

г
9Generator/second_layer/fully_connected/kernel/Adam/AssignAssign2Generator/second_layer/fully_connected/kernel/AdamDGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ф
7Generator/second_layer/fully_connected/kernel/Adam/readIdentity2Generator/second_layer/fully_connected/kernel/Adam*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

щ
VGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
г
LGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ѓ
FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillVGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
*
T0
ю
4Generator/second_layer/fully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:

й
;Generator/second_layer/fully_connected/kernel/Adam_1/AssignAssign4Generator/second_layer/fully_connected/kernel/Adam_1FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(
ш
9Generator/second_layer/fully_connected/kernel/Adam_1/readIdentity4Generator/second_layer/fully_connected/kernel/Adam_1*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

б
BGenerator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
о
0Generator/second_layer/fully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:
Ц
7Generator/second_layer/fully_connected/bias/Adam/AssignAssign0Generator/second_layer/fully_connected/bias/AdamBGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
й
5Generator/second_layer/fully_connected/bias/Adam/readIdentity0Generator/second_layer/fully_connected/bias/Adam*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:*
T0
г
DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    
р
2Generator/second_layer/fully_connected/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias
Ь
9Generator/second_layer/fully_connected/bias/Adam_1/AssignAssign2Generator/second_layer/fully_connected/bias/Adam_1DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
н
7Generator/second_layer/fully_connected/bias/Adam_1/readIdentity2Generator/second_layer/fully_connected/bias/Adam_1*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:
л
GGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ш
5Generator/second_layer/batch_normalization/gamma/Adam
VariableV2*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
к
<Generator/second_layer/batch_normalization/gamma/Adam/AssignAssign5Generator/second_layer/batch_normalization/gamma/AdamGGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
ш
:Generator/second_layer/batch_normalization/gamma/Adam/readIdentity5Generator/second_layer/batch_normalization/gamma/Adam*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:
н
IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*    
ъ
7Generator/second_layer/batch_normalization/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:
р
>Generator/second_layer/batch_normalization/gamma/Adam_1/AssignAssign7Generator/second_layer/batch_normalization/gamma/Adam_1IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(
ь
<Generator/second_layer/batch_normalization/gamma/Adam_1/readIdentity7Generator/second_layer/batch_normalization/gamma/Adam_1*
_output_shapes	
:*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
й
FGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ц
4Generator/second_layer/batch_normalization/beta/Adam
VariableV2*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:*
dtype0
ж
;Generator/second_layer/batch_normalization/beta/Adam/AssignAssign4Generator/second_layer/batch_normalization/beta/AdamFGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
х
9Generator/second_layer/batch_normalization/beta/Adam/readIdentity4Generator/second_layer/batch_normalization/beta/Adam*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:
л
HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ш
6Generator/second_layer/batch_normalization/beta/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
м
=Generator/second_layer/batch_normalization/beta/Adam_1/AssignAssign6Generator/second_layer/batch_normalization/beta/Adam_1HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
щ
;Generator/second_layer/batch_normalization/beta/Adam_1/readIdentity6Generator/second_layer/batch_normalization/beta/Adam_1*
_output_shapes	
:*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
х
SGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0
Я
IGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
щ
CGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ъ
1Generator/third_layer/fully_connected/kernel/Adam
VariableV2*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Я
8Generator/third_layer/fully_connected/kernel/Adam/AssignAssign1Generator/third_layer/fully_connected/kernel/AdamCGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
с
6Generator/third_layer/fully_connected/kernel/Adam/readIdentity1Generator/third_layer/fully_connected/kernel/Adam*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
ч
UGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
б
KGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
я
EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0
ь
3Generator/third_layer/fully_connected/kernel/Adam_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
е
:Generator/third_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/third_layer/fully_connected/kernel/Adam_1EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

х
8Generator/third_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/third_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

Я
AGenerator/third_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
м
/Generator/third_layer/fully_connected/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container 
Т
6Generator/third_layer/fully_connected/bias/Adam/AssignAssign/Generator/third_layer/fully_connected/bias/AdamAGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ж
4Generator/third_layer/fully_connected/bias/Adam/readIdentity/Generator/third_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:
б
CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
о
1Generator/third_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:
Ш
8Generator/third_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/third_layer/fully_connected/bias/Adam_1CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
к
6Generator/third_layer/fully_connected/bias/Adam_1/readIdentity1Generator/third_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:
й
FGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ц
4Generator/third_layer/batch_normalization/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:
ж
;Generator/third_layer/batch_normalization/gamma/Adam/AssignAssign4Generator/third_layer/batch_normalization/gamma/AdamFGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
х
9Generator/third_layer/batch_normalization/gamma/Adam/readIdentity4Generator/third_layer/batch_normalization/gamma/Adam*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
л
HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*    *
dtype0
ш
6Generator/third_layer/batch_normalization/gamma/Adam_1
VariableV2*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
м
=Generator/third_layer/batch_normalization/gamma/Adam_1/AssignAssign6Generator/third_layer/batch_normalization/gamma/Adam_1HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
щ
;Generator/third_layer/batch_normalization/gamma/Adam_1/readIdentity6Generator/third_layer/batch_normalization/gamma/Adam_1*
_output_shapes	
:*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma
з
EGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    
ф
3Generator/third_layer/batch_normalization/beta/Adam
VariableV2*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:*
dtype0
в
:Generator/third_layer/batch_normalization/beta/Adam/AssignAssign3Generator/third_layer/batch_normalization/beta/AdamEGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
т
8Generator/third_layer/batch_normalization/beta/Adam/readIdentity3Generator/third_layer/batch_normalization/beta/Adam*
_output_shapes	
:*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
й
GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ц
5Generator/third_layer/batch_normalization/beta/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:
и
<Generator/third_layer/batch_normalization/beta/Adam_1/AssignAssign5Generator/third_layer/batch_normalization/beta/Adam_1GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
ц
:Generator/third_layer/batch_normalization/beta/Adam_1/readIdentity5Generator/third_layer/batch_normalization/beta/Adam_1*
_output_shapes	
:*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
у
RGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0
Э
HGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
х
BGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zerosFillRGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ш
0Generator/last_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:

Ы
7Generator/last_layer/fully_connected/kernel/Adam/AssignAssign0Generator/last_layer/fully_connected/kernel/AdamBGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

о
5Generator/last_layer/fully_connected/kernel/Adam/readIdentity0Generator/last_layer/fully_connected/kernel/Adam*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

х
TGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Я
JGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ы
DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillTGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorJGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0
ъ
2Generator/last_layer/fully_connected/kernel/Adam_1
VariableV2*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
б
9Generator/last_layer/fully_connected/kernel/Adam_1/AssignAssign2Generator/last_layer/fully_connected/kernel/Adam_1DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
т
7Generator/last_layer/fully_connected/kernel/Adam_1/readIdentity2Generator/last_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
й
PGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0
Щ
FGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
и
@Generator/last_layer/fully_connected/bias/Adam/Initializer/zerosFillPGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorFGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:
к
.Generator/last_layer/fully_connected/bias/Adam
VariableV2*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
О
5Generator/last_layer/fully_connected/bias/Adam/AssignAssign.Generator/last_layer/fully_connected/bias/Adam@Generator/last_layer/fully_connected/bias/Adam/Initializer/zeros*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
г
3Generator/last_layer/fully_connected/bias/Adam/readIdentity.Generator/last_layer/fully_connected/bias/Adam*
_output_shapes	
:*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
л
RGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
Ы
HGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
о
BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zerosFillRGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/Const*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:*
T0
м
0Generator/last_layer/fully_connected/bias/Adam_1
VariableV2*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ф
7Generator/last_layer/fully_connected/bias/Adam_1/AssignAssign0Generator/last_layer/fully_connected/bias/Adam_1BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
з
5Generator/last_layer/fully_connected/bias/Adam_1/readIdentity0Generator/last_layer/fully_connected/bias/Adam_1*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:
у
UGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0
г
KGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ь
EGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zerosFillUGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorKGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/Const*
_output_shapes	
:*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
ф
3Generator/last_layer/batch_normalization/gamma/Adam
VariableV2*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
в
:Generator/last_layer/batch_normalization/gamma/Adam/AssignAssign3Generator/last_layer/batch_normalization/gamma/AdamEGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(
т
8Generator/last_layer/batch_normalization/gamma/Adam/readIdentity3Generator/last_layer/batch_normalization/gamma/Adam*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:*
T0
х
WGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0*
_output_shapes
:
е
MGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ђ
GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zerosFillWGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorMGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
ц
5Generator/last_layer/batch_normalization/gamma/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container 
и
<Generator/last_layer/batch_normalization/gamma/Adam_1/AssignAssign5Generator/last_layer/batch_normalization/gamma/Adam_1GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
ц
:Generator/last_layer/batch_normalization/gamma/Adam_1/readIdentity5Generator/last_layer/batch_normalization/gamma/Adam_1*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:
с
TGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:*
dtype0
б
JGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ш
DGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zerosFillTGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorJGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:
т
2Generator/last_layer/batch_normalization/beta/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
Ю
9Generator/last_layer/batch_normalization/beta/Adam/AssignAssign2Generator/last_layer/batch_normalization/beta/AdamDGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
п
7Generator/last_layer/batch_normalization/beta/Adam/readIdentity2Generator/last_layer/batch_normalization/beta/Adam*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:
у
VGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:*
dtype0*
_output_shapes
:
г
LGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ю
FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zerosFillVGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:
ф
4Generator/last_layer/batch_normalization/beta/Adam_1
VariableV2*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
д
;Generator/last_layer/batch_normalization/beta/Adam_1/AssignAssign4Generator/last_layer/batch_normalization/beta/Adam_1FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
у
9Generator/last_layer/batch_normalization/beta/Adam_1/readIdentity4Generator/last_layer/batch_normalization/beta/Adam_1*
_output_shapes	
:*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
У
BGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
­
8Generator/fake_image/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѕ
2Generator/fake_image/kernel/Adam/Initializer/zerosFillBGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensor8Generator/fake_image/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:

Ш
 Generator/fake_image/kernel/Adam
VariableV2*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:


'Generator/fake_image/kernel/Adam/AssignAssign Generator/fake_image/kernel/Adam2Generator/fake_image/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:

Ў
%Generator/fake_image/kernel/Adam/readIdentity Generator/fake_image/kernel/Adam* 
_output_shapes
:
*
T0*.
_class$
" loc:@Generator/fake_image/kernel
Х
DGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     
Џ
:Generator/fake_image/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ћ
4Generator/fake_image/kernel/Adam_1/Initializer/zerosFillDGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensor:Generator/fake_image/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0
Ъ
"Generator/fake_image/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:


)Generator/fake_image/kernel/Adam_1/AssignAssign"Generator/fake_image/kernel/Adam_14Generator/fake_image/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel
В
'Generator/fake_image/kernel/Adam_1/readIdentity"Generator/fake_image/kernel/Adam_1* 
_output_shapes
:
*
T0*.
_class$
" loc:@Generator/fake_image/kernel
­
0Generator/fake_image/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0*
_output_shapes	
:
К
Generator/fake_image/bias/Adam
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ў
%Generator/fake_image/bias/Adam/AssignAssignGenerator/fake_image/bias/Adam0Generator/fake_image/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias
Ѓ
#Generator/fake_image/bias/Adam/readIdentityGenerator/fake_image/bias/Adam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
Џ
2Generator/fake_image/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0*
_output_shapes	
:
М
 Generator/fake_image/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:

'Generator/fake_image/bias/Adam_1/AssignAssign Generator/fake_image/bias/Adam_12Generator/fake_image/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
Ї
%Generator/fake_image/bias/Adam_1/readIdentity Generator/fake_image/bias/Adam_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
Y
Adam_1/learning_rateConst*
valueB
 *ЗQ9*
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
 *wО?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
Н
DAdam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/first_layer/fully_connected/kernel1Generator/first_layer/fully_connected/kernel/Adam3Generator/first_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	d
А
BAdam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/first_layer/fully_connected/bias/Generator/first_layer/fully_connected/bias/Adam1Generator/first_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:
Ф
EAdam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam-Generator/second_layer/fully_connected/kernel2Generator/second_layer/fully_connected/kernel/Adam4Generator/second_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
Ж
CAdam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam+Generator/second_layer/fully_connected/bias0Generator/second_layer/fully_connected/bias/Adam2Generator/second_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:
й
HAdam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam0Generator/second_layer/batch_normalization/gamma5Generator/second_layer/batch_normalization/gamma/Adam7Generator/second_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
в
GAdam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam/Generator/second_layer/batch_normalization/beta4Generator/second_layer/batch_normalization/beta/Adam6Generator/second_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
О
DAdam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/third_layer/fully_connected/kernel1Generator/third_layer/fully_connected/kernel/Adam3Generator/third_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:

А
BAdam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/third_layer/fully_connected/bias/Generator/third_layer/fully_connected/bias/Adam1Generator/third_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:
г
GAdam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam/Generator/third_layer/batch_normalization/gamma4Generator/third_layer/batch_normalization/gamma/Adam6Generator/third_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:
Ь
FAdam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdam	ApplyAdam.Generator/third_layer/batch_normalization/beta3Generator/third_layer/batch_normalization/beta/Adam5Generator/third_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
И
CAdam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Generator/last_layer/fully_connected/kernel0Generator/last_layer/fully_connected/kernel/Adam2Generator/last_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:

Њ
AAdam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Generator/last_layer/fully_connected/bias.Generator/last_layer/fully_connected/bias/Adam0Generator/last_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
Э
FAdam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Generator/last_layer/batch_normalization/gamma3Generator/last_layer/batch_normalization/gamma/Adam5Generator/last_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Ц
EAdam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Generator/last_layer/batch_normalization/beta2Generator/last_layer/batch_normalization/beta/Adam4Generator/last_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:*
use_locking( 
и
3Adam_1/update_Generator/fake_image/kernel/ApplyAdam	ApplyAdamGenerator/fake_image/kernel Generator/fake_image/kernel/Adam"Generator/fake_image/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*.
_class$
" loc:@Generator/fake_image/kernel*
use_nesterov( * 
_output_shapes
:

Ъ
1Adam_1/update_Generator/fake_image/bias/ApplyAdam	ApplyAdamGenerator/fake_image/biasGenerator/fake_image/bias/Adam Generator/fake_image/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
use_nesterov( *
_output_shapes	
:
е	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
Њ
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
з	
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta22^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
Ў
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: ""7
	summaries*
(
discriminator_loss:0
generator_loss:0"к%
trainable_variablesТ%П%
ч
.Generator/first_layer/fully_connected/kernel:03Generator/first_layer/fully_connected/kernel/Assign3Generator/first_layer/fully_connected/kernel/read:02IGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
ж
,Generator/first_layer/fully_connected/bias:01Generator/first_layer/fully_connected/bias/Assign1Generator/first_layer/fully_connected/bias/read:02>Generator/first_layer/fully_connected/bias/Initializer/zeros:08
ы
/Generator/second_layer/fully_connected/kernel:04Generator/second_layer/fully_connected/kernel/Assign4Generator/second_layer/fully_connected/kernel/read:02JGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
к
-Generator/second_layer/fully_connected/bias:02Generator/second_layer/fully_connected/bias/Assign2Generator/second_layer/fully_connected/bias/read:02?Generator/second_layer/fully_connected/bias/Initializer/zeros:08
э
2Generator/second_layer/batch_normalization/gamma:07Generator/second_layer/batch_normalization/gamma/Assign7Generator/second_layer/batch_normalization/gamma/read:02CGenerator/second_layer/batch_normalization/gamma/Initializer/ones:08
ъ
1Generator/second_layer/batch_normalization/beta:06Generator/second_layer/batch_normalization/beta/Assign6Generator/second_layer/batch_normalization/beta/read:02CGenerator/second_layer/batch_normalization/beta/Initializer/zeros:08
ч
.Generator/third_layer/fully_connected/kernel:03Generator/third_layer/fully_connected/kernel/Assign3Generator/third_layer/fully_connected/kernel/read:02IGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform:08
ж
,Generator/third_layer/fully_connected/bias:01Generator/third_layer/fully_connected/bias/Assign1Generator/third_layer/fully_connected/bias/read:02>Generator/third_layer/fully_connected/bias/Initializer/zeros:08
щ
1Generator/third_layer/batch_normalization/gamma:06Generator/third_layer/batch_normalization/gamma/Assign6Generator/third_layer/batch_normalization/gamma/read:02BGenerator/third_layer/batch_normalization/gamma/Initializer/ones:08
ц
0Generator/third_layer/batch_normalization/beta:05Generator/third_layer/batch_normalization/beta/Assign5Generator/third_layer/batch_normalization/beta/read:02BGenerator/third_layer/batch_normalization/beta/Initializer/zeros:08
у
-Generator/last_layer/fully_connected/kernel:02Generator/last_layer/fully_connected/kernel/Assign2Generator/last_layer/fully_connected/kernel/read:02HGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform:08
в
+Generator/last_layer/fully_connected/bias:00Generator/last_layer/fully_connected/bias/Assign0Generator/last_layer/fully_connected/bias/read:02=Generator/last_layer/fully_connected/bias/Initializer/zeros:08
х
0Generator/last_layer/batch_normalization/gamma:05Generator/last_layer/batch_normalization/gamma/Assign5Generator/last_layer/batch_normalization/gamma/read:02AGenerator/last_layer/batch_normalization/gamma/Initializer/ones:08
т
/Generator/last_layer/batch_normalization/beta:04Generator/last_layer/batch_normalization/beta/Assign4Generator/last_layer/batch_normalization/beta/read:02AGenerator/last_layer/batch_normalization/beta/Initializer/zeros:08
Ѓ
Generator/fake_image/kernel:0"Generator/fake_image/kernel/Assign"Generator/fake_image/kernel/read:028Generator/fake_image/kernel/Initializer/random_uniform:08

Generator/fake_image/bias:0 Generator/fake_image/bias/Assign Generator/fake_image/bias/read:02-Generator/fake_image/bias/Initializer/zeros:08
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
Discriminator/prob/bias:0Discriminator/prob/bias/AssignDiscriminator/prob/bias/read:02+Discriminator/prob/bias/Initializer/zeros:08"
train_op

Adam
Adam_1"Е
	variablesІЂ
ч
.Generator/first_layer/fully_connected/kernel:03Generator/first_layer/fully_connected/kernel/Assign3Generator/first_layer/fully_connected/kernel/read:02IGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
ж
,Generator/first_layer/fully_connected/bias:01Generator/first_layer/fully_connected/bias/Assign1Generator/first_layer/fully_connected/bias/read:02>Generator/first_layer/fully_connected/bias/Initializer/zeros:08
ы
/Generator/second_layer/fully_connected/kernel:04Generator/second_layer/fully_connected/kernel/Assign4Generator/second_layer/fully_connected/kernel/read:02JGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
к
-Generator/second_layer/fully_connected/bias:02Generator/second_layer/fully_connected/bias/Assign2Generator/second_layer/fully_connected/bias/read:02?Generator/second_layer/fully_connected/bias/Initializer/zeros:08
э
2Generator/second_layer/batch_normalization/gamma:07Generator/second_layer/batch_normalization/gamma/Assign7Generator/second_layer/batch_normalization/gamma/read:02CGenerator/second_layer/batch_normalization/gamma/Initializer/ones:08
ъ
1Generator/second_layer/batch_normalization/beta:06Generator/second_layer/batch_normalization/beta/Assign6Generator/second_layer/batch_normalization/beta/read:02CGenerator/second_layer/batch_normalization/beta/Initializer/zeros:08

8Generator/second_layer/batch_normalization/moving_mean:0=Generator/second_layer/batch_normalization/moving_mean/Assign=Generator/second_layer/batch_normalization/moving_mean/read:02JGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros:0

<Generator/second_layer/batch_normalization/moving_variance:0AGenerator/second_layer/batch_normalization/moving_variance/AssignAGenerator/second_layer/batch_normalization/moving_variance/read:02MGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones:0
ч
.Generator/third_layer/fully_connected/kernel:03Generator/third_layer/fully_connected/kernel/Assign3Generator/third_layer/fully_connected/kernel/read:02IGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform:08
ж
,Generator/third_layer/fully_connected/bias:01Generator/third_layer/fully_connected/bias/Assign1Generator/third_layer/fully_connected/bias/read:02>Generator/third_layer/fully_connected/bias/Initializer/zeros:08
щ
1Generator/third_layer/batch_normalization/gamma:06Generator/third_layer/batch_normalization/gamma/Assign6Generator/third_layer/batch_normalization/gamma/read:02BGenerator/third_layer/batch_normalization/gamma/Initializer/ones:08
ц
0Generator/third_layer/batch_normalization/beta:05Generator/third_layer/batch_normalization/beta/Assign5Generator/third_layer/batch_normalization/beta/read:02BGenerator/third_layer/batch_normalization/beta/Initializer/zeros:08

7Generator/third_layer/batch_normalization/moving_mean:0<Generator/third_layer/batch_normalization/moving_mean/Assign<Generator/third_layer/batch_normalization/moving_mean/read:02IGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros:0

;Generator/third_layer/batch_normalization/moving_variance:0@Generator/third_layer/batch_normalization/moving_variance/Assign@Generator/third_layer/batch_normalization/moving_variance/read:02LGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones:0
у
-Generator/last_layer/fully_connected/kernel:02Generator/last_layer/fully_connected/kernel/Assign2Generator/last_layer/fully_connected/kernel/read:02HGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform:08
в
+Generator/last_layer/fully_connected/bias:00Generator/last_layer/fully_connected/bias/Assign0Generator/last_layer/fully_connected/bias/read:02=Generator/last_layer/fully_connected/bias/Initializer/zeros:08
х
0Generator/last_layer/batch_normalization/gamma:05Generator/last_layer/batch_normalization/gamma/Assign5Generator/last_layer/batch_normalization/gamma/read:02AGenerator/last_layer/batch_normalization/gamma/Initializer/ones:08
т
/Generator/last_layer/batch_normalization/beta:04Generator/last_layer/batch_normalization/beta/Assign4Generator/last_layer/batch_normalization/beta/read:02AGenerator/last_layer/batch_normalization/beta/Initializer/zeros:08
ќ
6Generator/last_layer/batch_normalization/moving_mean:0;Generator/last_layer/batch_normalization/moving_mean/Assign;Generator/last_layer/batch_normalization/moving_mean/read:02HGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros:0

:Generator/last_layer/batch_normalization/moving_variance:0?Generator/last_layer/batch_normalization/moving_variance/Assign?Generator/last_layer/batch_normalization/moving_variance/read:02KGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones:0
Ѓ
Generator/fake_image/kernel:0"Generator/fake_image/kernel/Assign"Generator/fake_image/kernel/read:028Generator/fake_image/kernel/Initializer/random_uniform:08

Generator/fake_image/bias:0 Generator/fake_image/bias/Assign Generator/fake_image/bias/read:02-Generator/fake_image/bias/Initializer/zeros:08
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

7Discriminator/first_layer/fully_connected/kernel/Adam:0<Discriminator/first_layer/fully_connected/kernel/Adam/Assign<Discriminator/first_layer/fully_connected/kernel/Adam/read:02IDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros:0

9Discriminator/first_layer/fully_connected/kernel/Adam_1:0>Discriminator/first_layer/fully_connected/kernel/Adam_1/Assign>Discriminator/first_layer/fully_connected/kernel/Adam_1/read:02KDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ј
5Discriminator/first_layer/fully_connected/bias/Adam:0:Discriminator/first_layer/fully_connected/bias/Adam/Assign:Discriminator/first_layer/fully_connected/bias/Adam/read:02GDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros:0

7Discriminator/first_layer/fully_connected/bias/Adam_1:0<Discriminator/first_layer/fully_connected/bias/Adam_1/Assign<Discriminator/first_layer/fully_connected/bias/Adam_1/read:02IDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros:0

8Discriminator/second_layer/fully_connected/kernel/Adam:0=Discriminator/second_layer/fully_connected/kernel/Adam/Assign=Discriminator/second_layer/fully_connected/kernel/Adam/read:02JDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros:0

:Discriminator/second_layer/fully_connected/kernel/Adam_1:0?Discriminator/second_layer/fully_connected/kernel/Adam_1/Assign?Discriminator/second_layer/fully_connected/kernel/Adam_1/read:02LDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ќ
6Discriminator/second_layer/fully_connected/bias/Adam:0;Discriminator/second_layer/fully_connected/bias/Adam/Assign;Discriminator/second_layer/fully_connected/bias/Adam/read:02HDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros:0

8Discriminator/second_layer/fully_connected/bias/Adam_1:0=Discriminator/second_layer/fully_connected/bias/Adam_1/Assign=Discriminator/second_layer/fully_connected/bias/Adam_1/read:02JDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
Є
 Discriminator/prob/kernel/Adam:0%Discriminator/prob/kernel/Adam/Assign%Discriminator/prob/kernel/Adam/read:022Discriminator/prob/kernel/Adam/Initializer/zeros:0
Ќ
"Discriminator/prob/kernel/Adam_1:0'Discriminator/prob/kernel/Adam_1/Assign'Discriminator/prob/kernel/Adam_1/read:024Discriminator/prob/kernel/Adam_1/Initializer/zeros:0

Discriminator/prob/bias/Adam:0#Discriminator/prob/bias/Adam/Assign#Discriminator/prob/bias/Adam/read:020Discriminator/prob/bias/Adam/Initializer/zeros:0
Є
 Discriminator/prob/bias/Adam_1:0%Discriminator/prob/bias/Adam_1/Assign%Discriminator/prob/bias/Adam_1/read:022Discriminator/prob/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
№
3Generator/first_layer/fully_connected/kernel/Adam:08Generator/first_layer/fully_connected/kernel/Adam/Assign8Generator/first_layer/fully_connected/kernel/Adam/read:02EGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ј
5Generator/first_layer/fully_connected/kernel/Adam_1:0:Generator/first_layer/fully_connected/kernel/Adam_1/Assign:Generator/first_layer/fully_connected/kernel/Adam_1/read:02GGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ш
1Generator/first_layer/fully_connected/bias/Adam:06Generator/first_layer/fully_connected/bias/Adam/Assign6Generator/first_layer/fully_connected/bias/Adam/read:02CGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros:0
№
3Generator/first_layer/fully_connected/bias/Adam_1:08Generator/first_layer/fully_connected/bias/Adam_1/Assign8Generator/first_layer/fully_connected/bias/Adam_1/read:02EGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
є
4Generator/second_layer/fully_connected/kernel/Adam:09Generator/second_layer/fully_connected/kernel/Adam/Assign9Generator/second_layer/fully_connected/kernel/Adam/read:02FGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ќ
6Generator/second_layer/fully_connected/kernel/Adam_1:0;Generator/second_layer/fully_connected/kernel/Adam_1/Assign;Generator/second_layer/fully_connected/kernel/Adam_1/read:02HGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ь
2Generator/second_layer/fully_connected/bias/Adam:07Generator/second_layer/fully_connected/bias/Adam/Assign7Generator/second_layer/fully_connected/bias/Adam/read:02DGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros:0
є
4Generator/second_layer/fully_connected/bias/Adam_1:09Generator/second_layer/fully_connected/bias/Adam_1/Assign9Generator/second_layer/fully_connected/bias/Adam_1/read:02FGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros:0

7Generator/second_layer/batch_normalization/gamma/Adam:0<Generator/second_layer/batch_normalization/gamma/Adam/Assign<Generator/second_layer/batch_normalization/gamma/Adam/read:02IGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros:0

9Generator/second_layer/batch_normalization/gamma/Adam_1:0>Generator/second_layer/batch_normalization/gamma/Adam_1/Assign>Generator/second_layer/batch_normalization/gamma/Adam_1/read:02KGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
ќ
6Generator/second_layer/batch_normalization/beta/Adam:0;Generator/second_layer/batch_normalization/beta/Adam/Assign;Generator/second_layer/batch_normalization/beta/Adam/read:02HGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros:0

8Generator/second_layer/batch_normalization/beta/Adam_1:0=Generator/second_layer/batch_normalization/beta/Adam_1/Assign=Generator/second_layer/batch_normalization/beta/Adam_1/read:02JGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
№
3Generator/third_layer/fully_connected/kernel/Adam:08Generator/third_layer/fully_connected/kernel/Adam/Assign8Generator/third_layer/fully_connected/kernel/Adam/read:02EGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ј
5Generator/third_layer/fully_connected/kernel/Adam_1:0:Generator/third_layer/fully_connected/kernel/Adam_1/Assign:Generator/third_layer/fully_connected/kernel/Adam_1/read:02GGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ш
1Generator/third_layer/fully_connected/bias/Adam:06Generator/third_layer/fully_connected/bias/Adam/Assign6Generator/third_layer/fully_connected/bias/Adam/read:02CGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros:0
№
3Generator/third_layer/fully_connected/bias/Adam_1:08Generator/third_layer/fully_connected/bias/Adam_1/Assign8Generator/third_layer/fully_connected/bias/Adam_1/read:02EGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
ќ
6Generator/third_layer/batch_normalization/gamma/Adam:0;Generator/third_layer/batch_normalization/gamma/Adam/Assign;Generator/third_layer/batch_normalization/gamma/Adam/read:02HGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros:0

8Generator/third_layer/batch_normalization/gamma/Adam_1:0=Generator/third_layer/batch_normalization/gamma/Adam_1/Assign=Generator/third_layer/batch_normalization/gamma/Adam_1/read:02JGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
ј
5Generator/third_layer/batch_normalization/beta/Adam:0:Generator/third_layer/batch_normalization/beta/Adam/Assign:Generator/third_layer/batch_normalization/beta/Adam/read:02GGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros:0

7Generator/third_layer/batch_normalization/beta/Adam_1:0<Generator/third_layer/batch_normalization/beta/Adam_1/Assign<Generator/third_layer/batch_normalization/beta/Adam_1/read:02IGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
ь
2Generator/last_layer/fully_connected/kernel/Adam:07Generator/last_layer/fully_connected/kernel/Adam/Assign7Generator/last_layer/fully_connected/kernel/Adam/read:02DGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros:0
є
4Generator/last_layer/fully_connected/kernel/Adam_1:09Generator/last_layer/fully_connected/kernel/Adam_1/Assign9Generator/last_layer/fully_connected/kernel/Adam_1/read:02FGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ф
0Generator/last_layer/fully_connected/bias/Adam:05Generator/last_layer/fully_connected/bias/Adam/Assign5Generator/last_layer/fully_connected/bias/Adam/read:02BGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros:0
ь
2Generator/last_layer/fully_connected/bias/Adam_1:07Generator/last_layer/fully_connected/bias/Adam_1/Assign7Generator/last_layer/fully_connected/bias/Adam_1/read:02DGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
ј
5Generator/last_layer/batch_normalization/gamma/Adam:0:Generator/last_layer/batch_normalization/gamma/Adam/Assign:Generator/last_layer/batch_normalization/gamma/Adam/read:02GGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros:0

7Generator/last_layer/batch_normalization/gamma/Adam_1:0<Generator/last_layer/batch_normalization/gamma/Adam_1/Assign<Generator/last_layer/batch_normalization/gamma/Adam_1/read:02IGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
є
4Generator/last_layer/batch_normalization/beta/Adam:09Generator/last_layer/batch_normalization/beta/Adam/Assign9Generator/last_layer/batch_normalization/beta/Adam/read:02FGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros:0
ќ
6Generator/last_layer/batch_normalization/beta/Adam_1:0;Generator/last_layer/batch_normalization/beta/Adam_1/Assign;Generator/last_layer/batch_normalization/beta/Adam_1/read:02HGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
Ќ
"Generator/fake_image/kernel/Adam:0'Generator/fake_image/kernel/Adam/Assign'Generator/fake_image/kernel/Adam/read:024Generator/fake_image/kernel/Adam/Initializer/zeros:0
Д
$Generator/fake_image/kernel/Adam_1:0)Generator/fake_image/kernel/Adam_1/Assign)Generator/fake_image/kernel/Adam_1/read:026Generator/fake_image/kernel/Adam_1/Initializer/zeros:0
Є
 Generator/fake_image/bias/Adam:0%Generator/fake_image/bias/Adam/Assign%Generator/fake_image/bias/Adam/read:022Generator/fake_image/bias/Adam/Initializer/zeros:0
Ќ
"Generator/fake_image/bias/Adam_1:0'Generator/fake_image/bias/Adam_1/Assign'Generator/fake_image/bias/Adam_1/read:024Generator/fake_image/bias/Adam_1/Initializer/zeros:0А+SИњ       сN	[k7ъ§жA*ю
w
discriminator_loss*a	   `кЏѕ?   `кЏѕ?      №?!   `кЏѕ?)@z *e§?23?шЏ|ѕ?EЬРЂї?џџџџџџя:              №?        
s
generator_loss*a	   рІ~ц?   рІ~ц?      №?!   рІ~ц?)@ts5 п?2uoћpц?2gЧGќAш?џџџџџџя:              №?        Dгќ       Ъ{­	 sъ§жA(*ю
w
discriminator_loss*a	   Р@   Р@      №?!   Р@) iyШїy@2w`<f@ъ6vЃя@џџџџџџя:              №?        
s
generator_loss*a	   ;Дв?   ;Дв?      №?!   ;Дв?) DtнЕ?2Сб?_&AЛoДв?џџџџџџя:              №?        Pg2pќ       Ъ{­	Ит!ъ§жAP*ю
w
discriminator_loss*a	    Юњ?    Юњ?      №?!    Юњ?) ђВічt@2yLњтгџљ?SFiќ?џџџџџџя:              №?        
s
generator_loss*a	   ннг?   ннг?      №?!   ннг?) dъђЊИ?2_&AЛoДв?ЯCaДGд?џџџџџџя:              №?        чЂќ       Ъ{­	"ъ§жAx*ю
w
discriminator_loss*a	    !С?    !С?      №?!    !С?)@0~V?2!ЕЌзЛП?г8ЎsС?џџџџџџя:              №?        
s
generator_loss*a	   `јс@   `јс@      №?!   `јс@) бM8ё\@@2юїh:np@SЊІІпЎ@џџџџџџя:              №?        ЌпЙ§       bэD	W"ъ§жA *ю
w
discriminator_loss*a	    НК?    НК?      №?!    НК?) Т?2%gіcE9К?ЉЄ(!иМ?џџџџџџя:              №?        
s
generator_loss*a	   Рc@   Рc@      №?!   Рc@) їТH!5@2{2і.рл@!бv@џџџџџџя:              №?        (тФ§       bэD	HЫ"ъ§жAШ*ю
w
discriminator_loss*a	   XСБ?   XСБ?      №?!   XСБ?) љxќГs?2ЕяїУiHА?Ў]$AщБ?џџџџџџя:              №?        
s
generator_loss*a	   v @   v @      №?!   v @) Єms6@2!бv@иВvЉ5f@џџџџџџя:              №?        к^§       bэD	љwе"ъ§жA№*ю
w
discriminator_loss*a	    ыюК?    ыюК?      №?!    ыюК?) љВЛ6Ћ?2%gіcE9К?ЉЄ(!иМ?џџџџџџя:              №?        
s
generator_loss*a	   Рtm@   Рtm@      №?!   Рtm@)єЕA)@2uјrЪ­н@DKјЇ@џџџџџџя:              №?        ѓ?§       bэD	м}#ъ§жA*ю
w
discriminator_loss*a	   `ЯlІ?   `ЯlІ?      №?!   `ЯlІ?)@ЦBn_?2б/и*>І?ЭgЛwЈ?џџџџџџя:              №?        
s
generator_loss*a	   @`в@   @`в@      №?!   @`в@)ihЋF@@2юїh:np@SЊІІпЎ@џџџџџџя:              №?        ZГъ§       bэD	жtZ#ъ§жAР*ю
w
discriminator_loss*a	    CоЃ?    CоЃ?      №?!    CоЃ?)@вЕ1№ЋX?2ЫuSЭыaЂ?`Юлa8Є?џџџџџџя:              №?        
s
generator_loss*a	    	@    	@      №?!    	@)@ќc\pZ>@2иВvЉ5f@юїh:np@џџџџџџя:              №?        Vъ§       bэD	ф#ъ§жAш*ю
w
discriminator_loss*a	   к?   к?      №?!   к?) ГZR@?2Ю"uд?}Y4j?џџџџџџя:              №?        
s
generator_loss*a	   *@   *@      №?!   *@) cД>@2иВvЉ5f@юїh:np@џџџџџџя:              №?        &(§       bэD	ьJх#ъ§жA*ю
w
discriminator_loss*a	   `ФЏ?   `ФЏ?      №?!   `ФЏ?) ЄUm-?2к7c_XY?з#эh/?џџџџџџя:              №?        
s
generator_loss*a	   Р6i@   Р6i@      №?!   Р6i@)ЌЉ,B@2юїh:np@SЊІІпЎ@џџџџџџя:              №?        L=8q§       bэD	гJ,$ъ§жAИ*ю
w
discriminator_loss*a	   РЂ_?   РЂ_?      №?!   РЂ_?)Мuj'?2#Ї+(Х?к7c_XY?џџџџџџя:              №?        
s
generator_loss*a	   @ъХ@   @ъХ@      №?!   @ъХ@)ШцТD@2SЊІІпЎ@)ъаТ&@џџџџџџя:              №?        ПЭт§       bэD	Rw$ъ§жAр*ю
w
discriminator_loss*a	    ѓO?    ѓO?      №?!    ѓO?)@\Jrѕ4?2ъ Б&?RcУн?џџџџџџя:              №?        
s
generator_loss*a	    ж@    ж@      №?!    ж@)  $ТЉ8@2!бv@иВvЉ5f@џџџџџџя:              №?        cВВ§       bэD	лћХ$ъ§жA*ю
w
discriminator_loss*a	    ж,Ѓ?    ж,Ѓ?      №?!    ж,Ѓ?) @.фљњV?2ЫuSЭыaЂ?`Юлa8Є?џџџџџџя:              №?        
s
generator_loss*a	   @OЬ@   @OЬ@      №?!   @OЬ@) hfЯЂ1@2DKјЇ@{2і.рл@џџџџџџя:              №?        Ъ0Aл§       bэD	ТШ%ъ§жAА*ю
w
discriminator_loss*a	   РgзІ?   РgзІ?      №?!   РgзІ?)`tЃиM`?2б/и*>І?ЭgЛwЈ?џџџџџџя:              №?        
s
generator_loss*a	    Зи@    Зи@      №?!    Зи@) JЗO@@2юїh:np@SЊІІпЎ@џџџџџџя:              №?        (8p§       bэD	qz%ъ§жAи*ю
w
discriminator_loss*a	    ,Љ?    ,Љ?      №?!    ,Љ?) ђ%МЭc?2ЭgЛwЈ?ќОсg№щЊ?џџџџџџя:              №?        
s
generator_loss*a	   РЉ@   РЉ@      №?!   РЉ@)xcD@2SЊІІпЎ@)ъаТ&@џџџџџџя:              №?        6л§       bэD	Вsе%ъ§жA*ю
w
discriminator_loss*a	    Ъм?    Ъм?      №?!    Ъм?)  {ОКЙ/?2з#эh/?ъ Б&?џџџџџџя:              №?        
s
generator_loss*a	   @zR@   @zR@      №?!   @zR@)[рTG@2)ъаТ&@a/5Lжн@џџџџџџя:              №?        jV§       bэD	иу(&ъ§жAЈ*ю
w
discriminator_loss*a	   `њ?   `њ?      №?!   `њ?)@њ=йNo4?2ъ Б&?RcУн?џџџџџџя:              №?        
s
generator_loss*a	    Ъ@    Ъ@      №?!    Ъ@) Щ!F@2SЊІІпЎ@)ъаТ&@џџџџџџя:              №?        аўЗ§       bэD	B.}&ъ§жAа*ю
w
discriminator_loss*a	   РЋа?   РЋа?      №?!   РЋа?)аYL}ђ)?2к7c_XY?з#эh/?џџџџџџя:              №?        
s
generator_loss*a	   Р;@   Р;@      №?!   Р;@)bML@2a/5Lжн@v@н5m @џџџџџџя:              №?        #Д§       bэD	 Mд&ъ§жAј*ю
w
discriminator_loss*a	   рч"z?   рч"z?      №?!   рч"z?) 0LпоX?2*QHЅx?oЯ5sz?џџџџџџя:              №?        
s
generator_loss*a	   w@   w@      №?!   w@) МgG@2)ъаТ&@a/5Lжн@џџџџџџя:              №?        Ы0§       bэD	<e+'ъ§жA *ю
w
discriminator_loss*a	   
m?   
m?      №?!   
m?) ЫD[ъ>2пЄж(g%k?іNрWмm?џџџџџџя:              №?        
s
generator_loss*a	    фM@    фM@      №?!    фM@) 	I@2)ъаТ&@a/5Lжн@џџџџџџя:              №?        ЉЬ§       bэD	Юэ'ъ§жAШ*ю
w
discriminator_loss*a	   кђw?   кђw?      №?!   кђw?) ђЃ&Mь?2&bе
мu?*QHЅx?џџџџџџя:              №?        
s
generator_loss*a	    ЁЖ@    ЁЖ@      №?!    ЁЖ@) њA H@2)ъаТ&@a/5Lжн@џџџџџџя:              №?        ЙvЭњ       сN	Рu(ъ§жA*ю
w
discriminator_loss*a	    Џ/w?    Џ/w?      №?!    Џ/w?) f,ЬвЬ ?2&bе
мu?*QHЅx?џџџџџџя:              №?        
s
generator_loss*a	    /@    /@      №?!    /@)  №'bJ@2)ъаТ&@a/5Lжн@џџџџџџя:              №?        ЄCЄќ       Ъ{­	4N-)ъ§жA(*ю
w
discriminator_loss*a	   `>x?   `>x?      №?!   `>x?) еEъ?2*QHЅx?oЯ5sz?џџџџџџя:              №?        
s
generator_loss*a	    йЦ@    йЦ@      №?!    йЦ@) ЊM@2a/5Lжн@v@н5m @џџџџџџя:              №?        wќ       Ъ{­	(г)ъ§жAP*ю
w
discriminator_loss*a	   @?   @?      №?!   @?) ЉX5xV?2>	 ?хнў=?џџџџџџя:              №?        
s
generator_loss*a	   РП@   РП@      №?!   РП@) бSpm=@2иВvЉ5f@юїh:np@џџџџџџя:              №?        їЌzЯќ       Ъ{­	Л є)ъ§жAx*ю
w
discriminator_loss*a	   `ў?   `ў?      №?!   `ў?)@ё ]?2хнў=?уРўJн\?џџџџџџя:              №?        
s
generator_loss*a	   Р>-@   Р>-@      №?!   Р>-@)їDB@2юїh:np@SЊІІпЎ@џџџџџџя:              №?        tшa§       bэD	pZ*ъ§жA *ю
w
discriminator_loss*a	    Л0?    Л0?      №?!    Л0?)@2XfЦ>?2^ЇSНР?Ю"uд?џџџџџџя:              №?        
s
generator_loss*a	   @Ds@   @Ds@      №?!   @Ds@)+ПљL@2a/5Lжн@v@н5m @џџџџџџя:              №?        W§       bэD	h2М*ъ§жAШ*ю
w
discriminator_loss*a	   Р[Э?   Р[Э?      №?!   Р[Э?) !&1Z6?2ъ Б&?RcУн?џџџџџџя:              №?        
s
generator_loss*a	   @   @      №?!   @) "QйvD@2SЊІІпЎ@)ъаТ&@џџџџџџя:              №?        !АgФ§       bэD	ёп#+ъ§жA№*ю
w
discriminator_loss*a	    H?    H?      №?!    H?) ШЧўH?2зШ< A?vмЩab?џџџџџџя:              №?        
s
generator_loss*a	   Рь@   Рь@      №?!   Рь@) )пlf*3@2{2і.рл@!бv@џџџџџџя:              №?        ib]o§       bэD	u+ъ§жA*ю
w
discriminator_loss*a	    Ь7Е?    Ь7Е?      №?!    Ь7Е?)@0i:#|?2І{ ЈЧГГ? l(ЌЕ?џџџџџџя:              №?        
s
generator_loss*a	   `ќ@   `ќ@      №?!   `ќ@) ix@E@2SЊІІпЎ@)ъаТ&@џџџџџџя:              №?        ` й§       bэD	ћ+ъ§жAР*ю
w
discriminator_loss*a	   Рсx?   Рсx?      №?!   Рсx?)X_ЉFD?2}Y4j?зШ< A?џџџџџџя:              №?        
s
generator_loss*a	   Q@   Q@      №?!   Q@) юьН2@2{2і.рл@!бv@џџџџџџя:              №?        lе§       bэD	 вi,ъ§жAш*ю
w
discriminator_loss*a	   Р?   Р?      №?!   Р?) Yм>$g?2хнў=?уРўJн\?џџџџџџя:              №?        
s
generator_loss*a	   Nќ@   Nќ@      №?!   Nќ@)  ЕNAJ@2)ъаТ&@a/5Lжн@џџџџџџя:              №?        єoе§       bэD	Opп,ъ§жA*ю
w
discriminator_loss*a	    ёс?    ёс?      №?!    ёс?)@деkQA?2уРўJн\?-дБL?џџџџџџя:              №?        
s
generator_loss*a	   @uю@   @uю@      №?!   @uю@)ХЭVaH@2)ъаТ&@a/5Lжн@џџџџџџя:              №?        QR§§       bэD	ћP-ъ§жAИ*ю
w
discriminator_loss*a	   '7~?   '7~?      №?!   '7~?) ТјЩ?2ЫДЪT}?>	 ?џџџџџџя