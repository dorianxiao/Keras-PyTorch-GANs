       ŁK"	   ÉęýÖAbrain.Event:2/Ď	     äg˙	ůÉęýÖA"ö
u
Generator/noise_inPlaceholder*
shape:˙˙˙˙˙˙˙˙˙d*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
ß
MGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      
Ń
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&ž*
dtype0*
_output_shapes
: 
Ń
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&>*
dtype0*
_output_shapes
: 
Ć
UGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
seed2 *
dtype0*
_output_shapes
:	d*

seed 
Î
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
: 
á
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
Ó
GGenerator/first_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d*
T0
ă
,Generator/first_layer/fully_connected/kernel
VariableV2*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name 
Č
3Generator/first_layer/fully_connected/kernel/AssignAssign,Generator/first_layer/fully_connected/kernelGGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(
Ö
1Generator/first_layer/fully_connected/kernel/readIdentity,Generator/first_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
Ę
<Generator/first_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    
×
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
ł
1Generator/first_layer/fully_connected/bias/AssignAssign*Generator/first_layer/fully_connected/bias<Generator/first_layer/fully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
Ě
/Generator/first_layer/fully_connected/bias/readIdentity*Generator/first_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:
Ö
,Generator/first_layer/fully_connected/MatMulMatMulGenerator/noise_in1Generator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
á
-Generator/first_layer/fully_connected/BiasAddBiasAdd,Generator/first_layer/fully_connected/MatMul/Generator/first_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
k
&Generator/first_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
ľ
$Generator/first_layer/leaky_relu/mulMul&Generator/first_layer/leaky_relu/alpha-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
 Generator/first_layer/leaky_reluMaximum$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
NGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ó
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   ž*
dtype0*
_output_shapes
: 
Ó
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
Ę
VGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformNGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
seed2 
Ň
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
ć
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulVGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

Ř
HGenerator/second_layer/fully_connected/kernel/Initializer/random_uniformAddLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
ç
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
Í
4Generator/second_layer/fully_connected/kernel/AssignAssign-Generator/second_layer/fully_connected/kernelHGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

Ú
2Generator/second_layer/fully_connected/kernel/readIdentity-Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
Ě
=Generator/second_layer/fully_connected/bias/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ů
+Generator/second_layer/fully_connected/bias
VariableV2*
_output_shapes	
:*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0
ˇ
2Generator/second_layer/fully_connected/bias/AssignAssign+Generator/second_layer/fully_connected/bias=Generator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ď
0Generator/second_layer/fully_connected/bias/readIdentity+Generator/second_layer/fully_connected/bias*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:
ć
-Generator/second_layer/fully_connected/MatMulMatMul Generator/first_layer/leaky_relu2Generator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ä
.Generator/second_layer/fully_connected/BiasAddBiasAdd-Generator/second_layer/fully_connected/MatMul0Generator/second_layer/fully_connected/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
Ő
AGenerator/second_layer/batch_normalization/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*  ?
ă
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
Ę
7Generator/second_layer/batch_normalization/gamma/AssignAssign0Generator/second_layer/batch_normalization/gammaAGenerator/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
Ţ
5Generator/second_layer/batch_normalization/gamma/readIdentity0Generator/second_layer/batch_normalization/gamma*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:
Ô
AGenerator/second_layer/batch_normalization/beta/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0
á
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
Ç
6Generator/second_layer/batch_normalization/beta/AssignAssign/Generator/second_layer/batch_normalization/betaAGenerator/second_layer/batch_normalization/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
Ű
4Generator/second_layer/batch_normalization/beta/readIdentity/Generator/second_layer/batch_normalization/beta*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:
â
HGenerator/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ď
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
ă
=Generator/second_layer/batch_normalization/moving_mean/AssignAssign6Generator/second_layer/batch_normalization/moving_meanHGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
đ
;Generator/second_layer/batch_normalization/moving_mean/readIdentity6Generator/second_layer/batch_normalization/moving_mean*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
_output_shapes	
:
é
KGenerator/second_layer/batch_normalization/moving_variance/Initializer/onesConst*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
÷
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
ň
AGenerator/second_layer/batch_normalization/moving_variance/AssignAssign:Generator/second_layer/batch_normalization/moving_varianceKGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance
ü
?Generator/second_layer/batch_normalization/moving_variance/readIdentity:Generator/second_layer/batch_normalization/moving_variance*
_output_shapes	
:*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance

:Generator/second_layer/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o:*
dtype0
â
8Generator/second_layer/batch_normalization/batchnorm/addAdd?Generator/second_layer/batch_normalization/moving_variance/read:Generator/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:
Ł
:Generator/second_layer/batch_normalization/batchnorm/RsqrtRsqrt8Generator/second_layer/batch_normalization/batchnorm/add*
_output_shapes	
:*
T0
Ř
8Generator/second_layer/batch_normalization/batchnorm/mulMul:Generator/second_layer/batch_normalization/batchnorm/Rsqrt5Generator/second_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
Ţ
:Generator/second_layer/batch_normalization/batchnorm/mul_1Mul.Generator/second_layer/fully_connected/BiasAdd8Generator/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ţ
:Generator/second_layer/batch_normalization/batchnorm/mul_2Mul;Generator/second_layer/batch_normalization/moving_mean/read8Generator/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:*
T0
×
8Generator/second_layer/batch_normalization/batchnorm/subSub4Generator/second_layer/batch_normalization/beta/read:Generator/second_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:*
T0
ę
:Generator/second_layer/batch_normalization/batchnorm/add_1Add:Generator/second_layer/batch_normalization/batchnorm/mul_18Generator/second_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
'Generator/second_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Ä
%Generator/second_layer/leaky_relu/mulMul'Generator/second_layer/leaky_relu/alpha:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
!Generator/second_layer/leaky_reluMaximum%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ß
MGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ń
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *óľ˝*
dtype0*
_output_shapes
: 
Ń
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *óľ=*
dtype0*
_output_shapes
: 
Ç
UGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Î
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
_output_shapes
: 
â
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/sub*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
Ô
GGenerator/third_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

ĺ
,Generator/third_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:

É
3Generator/third_layer/fully_connected/kernel/AssignAssign,Generator/third_layer/fully_connected/kernelGGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
×
1Generator/third_layer/fully_connected/kernel/readIdentity,Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
Ę
<Generator/third_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
×
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
ł
1Generator/third_layer/fully_connected/bias/AssignAssign*Generator/third_layer/fully_connected/bias<Generator/third_layer/fully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
Ě
/Generator/third_layer/fully_connected/bias/readIdentity*Generator/third_layer/fully_connected/bias*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:*
T0
ĺ
,Generator/third_layer/fully_connected/MatMulMatMul!Generator/second_layer/leaky_relu1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
á
-Generator/third_layer/fully_connected/BiasAddBiasAdd,Generator/third_layer/fully_connected/MatMul/Generator/third_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
@Generator/third_layer/batch_normalization/gamma/Initializer/onesConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
á
/Generator/third_layer/batch_normalization/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:
Ć
6Generator/third_layer/batch_normalization/gamma/AssignAssign/Generator/third_layer/batch_normalization/gamma@Generator/third_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
Ű
4Generator/third_layer/batch_normalization/gamma/readIdentity/Generator/third_layer/batch_normalization/gamma*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
Ň
@Generator/third_layer/batch_normalization/beta/Initializer/zerosConst*
_output_shapes	
:*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    *
dtype0
ß
.Generator/third_layer/batch_normalization/beta
VariableV2*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ă
5Generator/third_layer/batch_normalization/beta/AssignAssign.Generator/third_layer/batch_normalization/beta@Generator/third_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
Ř
3Generator/third_layer/batch_normalization/beta/readIdentity.Generator/third_layer/batch_normalization/beta*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:
ŕ
GGenerator/third_layer/batch_normalization/moving_mean/Initializer/zerosConst*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
í
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
ß
<Generator/third_layer/batch_normalization/moving_mean/AssignAssign5Generator/third_layer/batch_normalization/moving_meanGGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:
í
:Generator/third_layer/batch_normalization/moving_mean/readIdentity5Generator/third_layer/batch_normalization/moving_mean*
_output_shapes	
:*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean
ç
JGenerator/third_layer/batch_normalization/moving_variance/Initializer/onesConst*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
ő
9Generator/third_layer/batch_normalization/moving_variance
VariableV2*
shared_name *L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:
î
@Generator/third_layer/batch_normalization/moving_variance/AssignAssign9Generator/third_layer/batch_normalization/moving_varianceJGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance
ů
>Generator/third_layer/batch_normalization/moving_variance/readIdentity9Generator/third_layer/batch_normalization/moving_variance*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
_output_shapes	
:*
T0
~
9Generator/third_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
ß
7Generator/third_layer/batch_normalization/batchnorm/addAdd>Generator/third_layer/batch_normalization/moving_variance/read9Generator/third_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:
Ą
9Generator/third_layer/batch_normalization/batchnorm/RsqrtRsqrt7Generator/third_layer/batch_normalization/batchnorm/add*
_output_shapes	
:*
T0
Ő
7Generator/third_layer/batch_normalization/batchnorm/mulMul9Generator/third_layer/batch_normalization/batchnorm/Rsqrt4Generator/third_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
Ű
9Generator/third_layer/batch_normalization/batchnorm/mul_1Mul-Generator/third_layer/fully_connected/BiasAdd7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
9Generator/third_layer/batch_normalization/batchnorm/mul_2Mul:Generator/third_layer/batch_normalization/moving_mean/read7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
Ô
7Generator/third_layer/batch_normalization/batchnorm/subSub3Generator/third_layer/batch_normalization/beta/read9Generator/third_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ç
9Generator/third_layer/batch_normalization/batchnorm/add_1Add9Generator/third_layer/batch_normalization/batchnorm/mul_17Generator/third_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
&Generator/third_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Á
$Generator/third_layer/leaky_relu/mulMul&Generator/third_layer/leaky_relu/alpha9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ż
 Generator/third_layer/leaky_reluMaximum$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
LGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ď
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  ˝*
dtype0*
_output_shapes
: 
Ď
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
Ä
TGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ę
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/subSubJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
_output_shapes
: 
Ţ
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

Đ
FGenerator/last_layer/fully_connected/kernel/Initializer/random_uniformAddJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

ă
+Generator/last_layer/fully_connected/kernel
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
Ĺ
2Generator/last_layer/fully_connected/kernel/AssignAssign+Generator/last_layer/fully_connected/kernelFGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(
Ô
0Generator/last_layer/fully_connected/kernel/readIdentity+Generator/last_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

Ô
KGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
Ä
AGenerator/last_layer/fully_connected/bias/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
É
;Generator/last_layer/fully_connected/bias/Initializer/zerosFillKGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorAGenerator/last_layer/fully_connected/bias/Initializer/zeros/Const*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:*
T0
Ő
)Generator/last_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:
Ż
0Generator/last_layer/fully_connected/bias/AssignAssign)Generator/last_layer/fully_connected/bias;Generator/last_layer/fully_connected/bias/Initializer/zeros*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
É
.Generator/last_layer/fully_connected/bias/readIdentity)Generator/last_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:
â
+Generator/last_layer/fully_connected/MatMulMatMul Generator/third_layer/leaky_relu0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
Ţ
,Generator/last_layer/fully_connected/BiasAddBiasAdd+Generator/last_layer/fully_connected/MatMul.Generator/last_layer/fully_connected/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
Ý
OGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0*
_output_shapes
:
Í
EGenerator/last_layer/batch_normalization/gamma/Initializer/ones/ConstConst*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *  ?*
dtype0
Ú
?Generator/last_layer/batch_normalization/gamma/Initializer/onesFillOGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorEGenerator/last_layer/batch_normalization/gamma/Initializer/ones/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:
ß
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
Â
5Generator/last_layer/batch_normalization/gamma/AssignAssign.Generator/last_layer/batch_normalization/gamma?Generator/last_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
Ř
3Generator/last_layer/batch_normalization/gamma/readIdentity.Generator/last_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:
Ü
OGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:*
dtype0
Ě
EGenerator/last_layer/batch_normalization/beta/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
Ů
?Generator/last_layer/batch_normalization/beta/Initializer/zerosFillOGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorEGenerator/last_layer/batch_normalization/beta/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:
Ý
-Generator/last_layer/batch_normalization/beta
VariableV2*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ż
4Generator/last_layer/batch_normalization/beta/AssignAssign-Generator/last_layer/batch_normalization/beta?Generator/last_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
Ő
2Generator/last_layer/batch_normalization/beta/readIdentity-Generator/last_layer/batch_normalization/beta*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:*
T0
ę
VGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB:*
dtype0*
_output_shapes
:
Ú
LGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
ő
FGenerator/last_layer/batch_normalization/moving_mean/Initializer/zerosFillVGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*

index_type0*
_output_shapes	
:
ë
4Generator/last_layer/batch_normalization/moving_mean
VariableV2*
shared_name *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ű
;Generator/last_layer/batch_normalization/moving_mean/AssignAssign4Generator/last_layer/batch_normalization/moving_meanFGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
ę
9Generator/last_layer/batch_normalization/moving_mean/readIdentity4Generator/last_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
_output_shapes	
:
ń
YGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB:*
dtype0*
_output_shapes
:
á
OGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB
 *  ?

IGenerator/last_layer/batch_normalization/moving_variance/Initializer/onesFillYGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorOGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/Const*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*

index_type0*
_output_shapes	
:*
T0
ó
8Generator/last_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
	container *
shape:
ę
?Generator/last_layer/batch_normalization/moving_variance/AssignAssign8Generator/last_layer/batch_normalization/moving_varianceIGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:
ö
=Generator/last_layer/batch_normalization/moving_variance/readIdentity8Generator/last_layer/batch_normalization/moving_variance*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
_output_shapes	
:*
T0
}
8Generator/last_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Ü
6Generator/last_layer/batch_normalization/batchnorm/addAdd=Generator/last_layer/batch_normalization/moving_variance/read8Generator/last_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:

8Generator/last_layer/batch_normalization/batchnorm/RsqrtRsqrt6Generator/last_layer/batch_normalization/batchnorm/add*
_output_shapes	
:*
T0
Ň
6Generator/last_layer/batch_normalization/batchnorm/mulMul8Generator/last_layer/batch_normalization/batchnorm/Rsqrt3Generator/last_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
Ř
8Generator/last_layer/batch_normalization/batchnorm/mul_1Mul,Generator/last_layer/fully_connected/BiasAdd6Generator/last_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ř
8Generator/last_layer/batch_normalization/batchnorm/mul_2Mul9Generator/last_layer/batch_normalization/moving_mean/read6Generator/last_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:*
T0
Ń
6Generator/last_layer/batch_normalization/batchnorm/subSub2Generator/last_layer/batch_normalization/beta/read8Generator/last_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ä
8Generator/last_layer/batch_normalization/batchnorm/add_1Add8Generator/last_layer/batch_normalization/batchnorm/mul_16Generator/last_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
%Generator/last_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
ž
#Generator/last_layer/leaky_relu/mulMul%Generator/last_layer/leaky_relu/alpha8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
Generator/last_layer/leaky_reluMaximum#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
<Generator/fake_image/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ż
:Generator/fake_image/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zők˝*
dtype0
Ż
:Generator/fake_image/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zők=*
dtype0*
_output_shapes
: 

DGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniformRandomUniform<Generator/fake_image/kernel/Initializer/random_uniform/shape*

seed *
T0*.
_class$
" loc:@Generator/fake_image/kernel*
seed2 *
dtype0* 
_output_shapes
:

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
6Generator/fake_image/kernel/Initializer/random_uniformAdd:Generator/fake_image/kernel/Initializer/random_uniform/mul:Generator/fake_image/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:

Ă
Generator/fake_image/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel
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
¤
 Generator/fake_image/kernel/readIdentityGenerator/fake_image/kernel*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:

¨
+Generator/fake_image/bias/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0*
_output_shapes	
:
ľ
Generator/fake_image/bias
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ď
 Generator/fake_image/bias/AssignAssignGenerator/fake_image/bias+Generator/fake_image/bias/Initializer/zeros*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

Generator/fake_image/bias/readIdentityGenerator/fake_image/bias*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:*
T0
Á
Generator/fake_image/MatMulMatMulGenerator/last_layer/leaky_relu Generator/fake_image/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
Ž
Generator/fake_image/BiasAddBiasAddGenerator/fake_image/MatMulGenerator/fake_image/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
Generator/fake_image/TanhTanhGenerator/fake_image/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
z
Discriminator/real_inPlaceholder*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
ç
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ů
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY˝*
dtype0*
_output_shapes
: 
Ů
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
Ó
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
seed2 *
dtype0
Ţ
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
ň
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
ä
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:

í
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
Ů
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ă
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
Ň
@Discriminator/first_layer/fully_connected/bias/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ß
.Discriminator/first_layer/fully_connected/bias
VariableV2*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0
Ă
5Discriminator/first_layer/fully_connected/bias/AssignAssign.Discriminator/first_layer/fully_connected/bias@Discriminator/first_layer/fully_connected/bias/Initializer/zeros*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ř
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
á
0Discriminator/first_layer/fully_connected/MatMulMatMulDiscriminator/real_in5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
í
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
*Discriminator/first_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Á
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
$Discriminator/first_layer/leaky_reluMaximum(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
é
RDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0
Ű
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *óľ˝*
dtype0*
_output_shapes
: 
Ű
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *óľ=*
dtype0*
_output_shapes
: 
Ö
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
â
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
_output_shapes
: 
ö
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

č
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

ď
1Discriminator/second_layer/fully_connected/kernel
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
Ý
8Discriminator/second_layer/fully_connected/kernel/AssignAssign1Discriminator/second_layer/fully_connected/kernelLDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

ć
6Discriminator/second_layer/fully_connected/kernel/readIdentity1Discriminator/second_layer/fully_connected/kernel*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

Ô
ADiscriminator/second_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    
á
/Discriminator/second_layer/fully_connected/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
Ç
6Discriminator/second_layer/fully_connected/bias/AssignAssign/Discriminator/second_layer/fully_connected/biasADiscriminator/second_layer/fully_connected/bias/Initializer/zeros*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ű
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:*
T0
ň
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
đ
2Discriminator/second_layer/fully_connected/BiasAddBiasAdd1Discriminator/second_layer/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
+Discriminator/second_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Ä
)Discriminator/second_layer/leaky_relu/mulMul+Discriminator/second_layer/leaky_relu/alpha2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
%Discriminator/second_layer/leaky_reluMaximum)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
:Discriminator/prob/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ť
8Discriminator/prob/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Ivž*
dtype0*
_output_shapes
: 
Ť
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
8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
: *
T0
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
˝
Discriminator/prob/kernel
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container 
ü
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
˘
)Discriminator/prob/bias/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
Ż
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
ć
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
Â
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
§
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ç
2Discriminator/first_layer_1/fully_connected/MatMulMatMulGenerator/fake_image/Tanh5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
ń
3Discriminator/first_layer_1/fully_connected/BiasAddBiasAdd2Discriminator/first_layer_1/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
,Discriminator/first_layer_1/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Ç
*Discriminator/first_layer_1/leaky_relu/mulMul,Discriminator/first_layer_1/leaky_relu/alpha3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
&Discriminator/first_layer_1/leaky_reluMaximum*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ö
3Discriminator/second_layer_1/fully_connected/MatMulMatMul&Discriminator/first_layer_1/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
ô
4Discriminator/second_layer_1/fully_connected/BiasAddBiasAdd3Discriminator/second_layer_1/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
-Discriminator/second_layer_1/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Ę
+Discriminator/second_layer_1/leaky_relu/mulMul-Discriminator/second_layer_1/leaky_relu/alpha4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
'Discriminator/second_layer_1/leaky_reluMaximum+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
Discriminator/prob_1/MatMulMatMul'Discriminator/second_layer_1/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
Ť
Discriminator/prob_1/BiasAddBiasAddDiscriminator/prob_1/MatMulDiscriminator/prob/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
ones_like/ShapeShapeDiscriminator/prob/BiasAdd*
out_type0*
_output_shapes
:*
T0
T
ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
w
	ones_likeFillones_like/Shapeones_like/Const*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
logistic_loss/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

logistic_loss/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
logistic_loss/SelectSelectlogistic_loss/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
logistic_loss/NegNegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/NegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
logistic_loss/mulMulDiscriminator/prob/BiasAdd	ones_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
logistic_loss/ExpExplogistic_loss/Select_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
`
MeanMeanlogistic_lossConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
g

zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
logistic_loss_1/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

logistic_loss_1/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ľ
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAdd
zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
f
logistic_loss_1/ExpExplogistic_loss_1/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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
ones_like_1/ShapeShapeDiscriminator/prob_1/BiasAdd*
_output_shapes
:*
T0*
out_type0
V
ones_like_1/ConstConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
}
ones_like_1Fillones_like_1/Shapeones_like_1/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
w
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
logistic_loss_2/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
w
logistic_loss_2/mulMulDiscriminator/prob_1/BiasAddones_like_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
logistic_loss_2/ExpExplogistic_loss_2/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
Const_2Const*
_output_shapes
:*
valueB"       *
dtype0
f
Mean_2Meanlogistic_loss_2Const_2*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
generator_loss/tagConst*
valueB Bgenerator_loss*
dtype0*
_output_shapes
: 
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
ą
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Fill$^gradients/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
ł
-gradients/add_grad/tuple/control_dependency_1Identitygradients/Fill$^gradients/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
r
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
­
gradients/Mean_grad/ReshapeReshape+gradients/add_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
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
:˙˙˙˙˙˙˙˙˙
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
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
:˙˙˙˙˙˙˙˙˙
t
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ł
gradients/Mean_1_grad/ReshapeReshape-gradients/add_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
j
gradients/Mean_1_grad/ShapeShapelogistic_loss_1*
T0*
out_type0*
_output_shapes
:
˘
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
l
gradients/Mean_1_grad/Shape_1Shapelogistic_loss_1*
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
gradients/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
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
:˙˙˙˙˙˙˙˙˙
s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0*
_output_shapes
:
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:
Ň
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
¸
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ľ
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ť
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1

5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape

7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
Ř
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ž
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ť
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Â
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Á
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/logistic_loss_1_grad/tuple/group_depsNoOp'^gradients/logistic_loss_1_grad/Reshape)^gradients/logistic_loss_1_grad/Reshape_1

7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
Ţ
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ú
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Á
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ţ
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
_output_shapes
:*
T0
Ĺ
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1

9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1
§
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
:˙˙˙˙˙˙˙˙˙

-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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
ä
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ŕ
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ç
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
(gradients/logistic_loss_1/sub_grad/Sum_1Sum7gradients/logistic_loss_1_grad/tuple/control_dependency:gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
z
&gradients/logistic_loss_1/sub_grad/NegNeg(gradients/logistic_loss_1/sub_grad/Sum_1*
T0*
_output_shapes
:
Ë
,gradients/logistic_loss_1/sub_grad/Reshape_1Reshape&gradients/logistic_loss_1/sub_grad/Neg*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

3gradients/logistic_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/sub_grad/Reshape-^gradients/logistic_loss_1/sub_grad/Reshape_1

;gradients/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/sub_grad/Reshape4^gradients/logistic_loss_1/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/sub_grad/Reshape
 
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
*gradients/logistic_loss_1/Log1p_grad/add/xConst:^gradients/logistic_loss_1_grad/tuple/control_dependency_1*
_output_shapes
: *
valueB
 *  ?*
dtype0
˘
(gradients/logistic_loss_1/Log1p_grad/addAdd*gradients/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

/gradients/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ď
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1

<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1

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
Ţ
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ľ
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Á
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
&gradients/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ç
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1

9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

0gradients/logistic_loss_1/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ő
,gradients/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual;gradients/logistic_loss_1/sub_grad/tuple/control_dependency0gradients/logistic_loss_1/Select_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
÷
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
¤
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
ä
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ş
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1
zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
&gradients/logistic_loss_1/mul_grad/SumSum&gradients/logistic_loss_1/mul_grad/Mul8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ç
*gradients/logistic_loss_1/mul_grad/ReshapeReshape&gradients/logistic_loss_1/mul_grad/Sum(gradients/logistic_loss_1/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
(gradients/logistic_loss_1/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ő
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Í
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

3gradients/logistic_loss_1/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/mul_grad/Reshape-^gradients/logistic_loss_1/mul_grad/Reshape_1

;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/mul_grad/Reshape4^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1

&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ţ
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
¤
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select
Ş
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
Ź
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1

$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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
:˙˙˙˙˙˙˙˙˙
ł
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ý
gradients/AddN_1AddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N

7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
data_formatNHWC*
_output_shapes
:*
T0

<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_18^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad

Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
ö
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
ö
1gradients/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
§
9gradients/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp0^gradients/Discriminator/prob/MatMul_grad/MatMul2^gradients/Discriminator/prob/MatMul_grad/MatMul_1
ą
Agradients/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity/gradients/Discriminator/prob/MatMul_grad/MatMul:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/Discriminator/prob/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	
ú
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ü
3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
­
;gradients/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp2^gradients/Discriminator/prob_1/MatMul_grad/MatMul4^gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
š
Cgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/Discriminator/prob_1/MatMul_grad/MatMul<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1

gradients/AddN_2AddNDgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
N
Ł
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ž
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
˝
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
:˙˙˙˙˙˙˙˙˙
ă
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Jgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Discriminator/second_layer/leaky_relu_grad/Shape<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ş
;gradients/Discriminator/second_layer/leaky_relu_grad/SelectSelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency:gradients/Discriminator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

8gradients/Discriminator/second_layer/leaky_relu_grad/SumSum;gradients/Discriminator/second_layer/leaky_relu_grad/SelectJgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ţ
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
Í
Egradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp=^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape?^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
ă
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
§
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
˛
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Á
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
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
:˙˙˙˙˙˙˙˙˙
é
Cgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
 
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Â
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape:gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
ë
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1

gradients/AddN_3AddNCgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
N

>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
˛
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ś
Ngradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ů
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulMulMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ř
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ô
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Pgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Bgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
Igradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpA^gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeC^gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
á
Qgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeJ^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
ů
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
ś
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ź
Pgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
˙
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ţ
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ú
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Rgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Dgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1Reshape@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ß
Kgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeE^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
é
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
:˙˙˙˙˙˙˙˙˙
Í
gradients/AddN_4AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:
˝
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ó
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ó
gradients/AddN_5AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:
Á
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
Ů
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ž
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ś
Igradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_reluZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ď
Qgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpH^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulJ^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1

Ygradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityGgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulR^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*Z
_classP
NLloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul

[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

Â
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ź
Kgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ő
Sgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulL^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1

[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulT^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

ç
gradients/AddN_6AddN\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
Ą
9gradients/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ź
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ô
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
:˙˙˙˙˙˙˙˙˙
ŕ
@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ď
:gradients/Discriminator/first_layer/leaky_relu_grad/SelectSelect@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency9gradients/Discriminator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ń
<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Select@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqual9gradients/Discriminator/first_layer/leaky_relu_grad/zerosYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

7gradients/Discriminator/first_layer/leaky_relu_grad/SumSum:gradients/Discriminator/first_layer/leaky_relu_grad/SelectIgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ű
;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape7gradients/Discriminator/first_layer/leaky_relu_grad/Sum9gradients/Discriminator/first_layer/leaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Kgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
Dgradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp<^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape>^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
ß
Lgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeE^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
°
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ř
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
;gradients/Discriminator/first_layer_1/leaky_relu_grad/zerosFill=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
×
<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ů
>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients/Discriminator/first_layer_1/leaky_relu_grad/SumSum<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectKgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Đ
Fgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp>^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape@^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
ç
Ngradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeG^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape
í
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
ć
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
°
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ł
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ö
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ő
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ń
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
Ö
Hgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp@^gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeB^gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
Ý
Pgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeI^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
ő
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
´
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Š
Ogradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ü
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulOgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ű
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
÷
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
Ü
Jgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeD^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
ĺ
Rgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeK^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
ý
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
Ę
gradients/AddN_8AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:
ť
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Đ
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradR^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Đ
gradients/AddN_9AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
data_formatNHWC*
_output_shapes	
:*
T0
ż
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
Ö
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ť
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(

Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/real_inYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ě
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1

Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
ż
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0

Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ň
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1

Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
ĺ
gradients/AddN_10AddN[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
ä
gradients/AddN_11AddNZgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:

Ą
beta1_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
˛
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
Ń
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias

beta1_power/readIdentitybeta1_power*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: *
T0
Ą
beta2_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *wž?*
dtype0*
_output_shapes
: 
˛
beta2_power
VariableV2*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Ń
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(

beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
í
WDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     
×
MDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ů
GDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillWDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorMDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
*
T0
ň
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
ß
<Discriminator/first_layer/fully_connected/kernel/Adam/AssignAssign5Discriminator/first_layer/fully_connected/kernel/AdamGDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
í
:Discriminator/first_layer/fully_connected/kernel/Adam/readIdentity5Discriminator/first_layer/fully_connected/kernel/Adam*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:

ď
YDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ů
ODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
˙
IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillYDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ô
7Discriminator/first_layer/fully_connected/kernel/Adam_1
VariableV2*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ĺ
>Discriminator/first_layer/fully_connected/kernel/Adam_1/AssignAssign7Discriminator/first_layer/fully_connected/kernel/Adam_1IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

ń
<Discriminator/first_layer/fully_connected/kernel/Adam_1/readIdentity7Discriminator/first_layer/fully_connected/kernel/Adam_1*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:

×
EDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ä
3Discriminator/first_layer/fully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:
Ň
:Discriminator/first_layer/fully_connected/bias/Adam/AssignAssign3Discriminator/first_layer/fully_connected/bias/AdamEDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
â
8Discriminator/first_layer/fully_connected/bias/Adam/readIdentity3Discriminator/first_layer/fully_connected/bias/Adam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:
Ů
GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    
ć
5Discriminator/first_layer/fully_connected/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container 
Ř
<Discriminator/first_layer/fully_connected/bias/Adam_1/AssignAssign5Discriminator/first_layer/fully_connected/bias/Adam_1GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ć
:Discriminator/first_layer/fully_connected/bias/Adam_1/readIdentity5Discriminator/first_layer/fully_connected/bias/Adam_1*
_output_shapes	
:*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
ď
XDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ů
NDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ý
HDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillXDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorNDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0
ô
6Discriminator/second_layer/fully_connected/kernel/Adam
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
ă
=Discriminator/second_layer/fully_connected/kernel/Adam/AssignAssign6Discriminator/second_layer/fully_connected/kernel/AdamHDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
đ
;Discriminator/second_layer/fully_connected/kernel/Adam/readIdentity6Discriminator/second_layer/fully_connected/kernel/Adam* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
ń
ZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ű
PDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorPDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ö
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
é
?Discriminator/second_layer/fully_connected/kernel/Adam_1/AssignAssign8Discriminator/second_layer/fully_connected/kernel/Adam_1JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
ô
=Discriminator/second_layer/fully_connected/kernel/Adam_1/readIdentity8Discriminator/second_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
Ů
FDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ć
4Discriminator/second_layer/fully_connected/bias/Adam
VariableV2*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ö
;Discriminator/second_layer/fully_connected/bias/Adam/AssignAssign4Discriminator/second_layer/fully_connected/bias/AdamFDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ĺ
9Discriminator/second_layer/fully_connected/bias/Adam/readIdentity4Discriminator/second_layer/fully_connected/bias/Adam*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
Ű
HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
č
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
Ü
=Discriminator/second_layer/fully_connected/bias/Adam_1/AssignAssign6Discriminator/second_layer/fully_connected/bias/Adam_1HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
é
;Discriminator/second_layer/fully_connected/bias/Adam_1/readIdentity6Discriminator/second_layer/fully_connected/bias/Adam_1*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
ľ
0Discriminator/prob/kernel/Adam/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Â
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
§
#Discriminator/prob/kernel/Adam/readIdentityDiscriminator/prob/kernel/Adam*
_output_shapes
:	*
T0*,
_class"
 loc:@Discriminator/prob/kernel
ˇ
2Discriminator/prob/kernel/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Ä
 Discriminator/prob/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	

'Discriminator/prob/kernel/Adam_1/AssignAssign Discriminator/prob/kernel/Adam_12Discriminator/prob/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	
Ť
%Discriminator/prob/kernel/Adam_1/readIdentity Discriminator/prob/kernel/Adam_1*
_output_shapes
:	*
T0*,
_class"
 loc:@Discriminator/prob/kernel
§
.Discriminator/prob/bias/Adam/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
´
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
ő
#Discriminator/prob/bias/Adam/AssignAssignDiscriminator/prob/bias/Adam.Discriminator/prob/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias

!Discriminator/prob/bias/Adam/readIdentityDiscriminator/prob/bias/Adam*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
Š
0Discriminator/prob/bias/Adam_1/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
ś
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
ű
%Discriminator/prob/bias/Adam_1/AssignAssignDiscriminator/prob/bias/Adam_10Discriminator/prob/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias
 
#Discriminator/prob/bias/Adam_1/readIdentityDiscriminator/prob/bias/Adam_1*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *ˇQ9*
dtype0
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
 *wž?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
ý
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
î
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
Adam/beta2Adam/epsilongradients/AddN_7*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:

ň
EAdam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/bias4Discriminator/second_layer/fully_connected/bias/Adam6Discriminator/second_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0

/Adam/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernelDiscriminator/prob/kernel/Adam Discriminator/prob/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
use_nesterov( *
_output_shapes
:	*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel
ů
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
Adam/beta1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: *
T0
š
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
˝
Adam/Assign_1Assignbeta2_power
Adam/mul_1*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
Ž
AdamNoOp^Adam/Assign^Adam/Assign_1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
gradients_1/Mean_2_grad/ShapeShapelogistic_loss_2*
_output_shapes
:*
T0*
out_type0
¨
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
gradients_1/Mean_2_grad/Shape_1Shapelogistic_loss_2*
out_type0*
_output_shapes
:*
T0
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
˘
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
Ś
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

gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0
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
:˙˙˙˙˙˙˙˙˙*
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
Ţ
6gradients_1/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_2_grad/Shape(gradients_1/logistic_loss_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ä
$gradients_1/logistic_loss_2_grad/SumSumgradients_1/Mean_2_grad/truediv6gradients_1/logistic_loss_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Á
(gradients_1/logistic_loss_2_grad/ReshapeReshape$gradients_1/logistic_loss_2_grad/Sum&gradients_1/logistic_loss_2_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Č
&gradients_1/logistic_loss_2_grad/Sum_1Sumgradients_1/Mean_2_grad/truediv8gradients_1/logistic_loss_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ç
*gradients_1/logistic_loss_2_grad/Reshape_1Reshape&gradients_1/logistic_loss_2_grad/Sum_1(gradients_1/logistic_loss_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1gradients_1/logistic_loss_2_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_2_grad/Reshape+^gradients_1/logistic_loss_2_grad/Reshape_1

9gradients_1/logistic_loss_2_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_2_grad/Reshape2^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_2_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients_1/logistic_loss_2_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_2_grad/Reshape_12^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss_2_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

*gradients_1/logistic_loss_2/sub_grad/ShapeShapelogistic_loss_2/Select*
T0*
out_type0*
_output_shapes
:

,gradients_1/logistic_loss_2/sub_grad/Shape_1Shapelogistic_loss_2/mul*
_output_shapes
:*
T0*
out_type0
ę
:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/sub_grad/Shape,gradients_1/logistic_loss_2/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ć
(gradients_1/logistic_loss_2/sub_grad/SumSum9gradients_1/logistic_loss_2_grad/tuple/control_dependency:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Í
,gradients_1/logistic_loss_2/sub_grad/ReshapeReshape(gradients_1/logistic_loss_2/sub_grad/Sum*gradients_1/logistic_loss_2/sub_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ę
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
Ń
.gradients_1/logistic_loss_2/sub_grad/Reshape_1Reshape(gradients_1/logistic_loss_2/sub_grad/Neg,gradients_1/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

5gradients_1/logistic_loss_2/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/sub_grad/Reshape/^gradients_1/logistic_loss_2/sub_grad/Reshape_1
˘
=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/sub_grad/Reshape6^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/sub_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/sub_grad/Reshape_16^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
,gradients_1/logistic_loss_2/Log1p_grad/add/xConst<^gradients_1/logistic_loss_2_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ś
*gradients_1/logistic_loss_2/Log1p_grad/addAdd,gradients_1/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1gradients_1/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_2/Log1p_grad/add*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ó
*gradients_1/logistic_loss_2/Log1p_grad/mulMul;gradients_1/logistic_loss_2_grad/tuple/control_dependency_11gradients_1/logistic_loss_2/Log1p_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

2gradients_1/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ű
.gradients_1/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_2/Select_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ý
0gradients_1/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients_1/logistic_loss_2/Select_grad/zeros_like=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¤
8gradients_1/logistic_loss_2/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_2/Select_grad/Select1^gradients_1/logistic_loss_2/Select_grad/Select_1
Ź
@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_2/Select_grad/Select9^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
Bgradients_1/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_2/Select_grad/Select_19^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
ę
:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/mul_grad/Shape,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ż
(gradients_1/logistic_loss_2/mul_grad/MulMul?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1ones_like_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
(gradients_1/logistic_loss_2/mul_grad/SumSum(gradients_1/logistic_loss_2/mul_grad/Mul:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Í
,gradients_1/logistic_loss_2/mul_grad/ReshapeReshape(gradients_1/logistic_loss_2/mul_grad/Sum*gradients_1/logistic_loss_2/mul_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Â
*gradients_1/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
*gradients_1/logistic_loss_2/mul_grad/Sum_1Sum*gradients_1/logistic_loss_2/mul_grad/Mul_1<gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ó
.gradients_1/logistic_loss_2/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_2/mul_grad/Sum_1,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

5gradients_1/logistic_loss_2/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/mul_grad/Reshape/^gradients_1/logistic_loss_2/mul_grad/Reshape_1
˘
=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/mul_grad/Reshape6^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
?gradients_1/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/mul_grad/Reshape_16^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
(gradients_1/logistic_loss_2/Exp_grad/mulMul*gradients_1/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

4gradients_1/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
0gradients_1/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual(gradients_1/logistic_loss_2/Exp_grad/mul4gradients_1/logistic_loss_2/Select_1_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
2gradients_1/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual4gradients_1/logistic_loss_2/Select_1_grad/zeros_like(gradients_1/logistic_loss_2/Exp_grad/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ş
:gradients_1/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_2/Select_1_grad/Select3^gradients_1/logistic_loss_2/Select_1_grad/Select_1
´
Bgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_2/Select_1_grad/Select;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_1_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
Dgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_2/Select_1_grad/Select_1;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/logistic_loss_2/Select_1_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
(gradients_1/logistic_loss_2/Neg_grad/NegNegBgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/AddNAddN@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_2/Neg_grad/Neg*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
data_formatNHWC*
_output_shapes
:*
T0

>gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN:^gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad

Fgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ă
Hgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
ţ
3gradients_1/Discriminator/prob_1/MatMul_grad/MatMulMatMulFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
ł
=gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul6^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1
Á
Egradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Discriminator/prob_1/MatMul_grad/MatMul>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
Ggradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*H
_class>
<:loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
Š
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
´
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ĺ
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
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
:˙˙˙˙˙˙˙˙˙
ë
Egradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
Ngradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ę
?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ě
Agradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SumSum?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectNgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1SumAgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1Pgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Bgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
Igradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOpA^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeC^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
ó
Qgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeJ^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
Sgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityBgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1J^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
¸
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
˛
Rgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulRgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ţ
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Tgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Fgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
Mgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpE^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeG^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
ń
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
:˙˙˙˙˙˙˙˙˙
Ű
gradients_1/AddN_1AddNSgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
Qgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
_output_shapes	
:*
T0*
data_formatNHWC
Ç
Vgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1R^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
á
^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1W^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¤
`gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityQgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ć
Kgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
°
Mgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ű
Ugradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpL^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulN^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
Ą
]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityKgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulV^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityMgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1V^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

§
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
˛
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ü
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
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
č
Dgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
Mgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ß
>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
á
@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SumSum>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectMgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
Agradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ö
Hgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp@^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeB^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
ď
Pgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeI^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape
ő
Rgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityAgradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1I^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1

Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
ś
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ż
Qgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulQgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
ű
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
 
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Sgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Egradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
Lgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpD^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeF^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
í
Tgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeM^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 

Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityEgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1M^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*X
_classN
LJloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
Ř
gradients_1/AddN_2AddNRgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
Pgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:
Ĺ
Ugradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2Q^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
Ţ
]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2V^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
 
_gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradV^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ă
Jgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ą
Lgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ř
Tgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpK^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulM^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1

\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityJgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulU^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul

^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityLgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1U^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

ë
3gradients_1/Generator/fake_image/Tanh_grad/TanhGradTanhGradGenerator/fake_image/Tanh\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:
¸
>gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad4^gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
Ă
Fgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
Ä
Hgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0

3gradients_1/Generator/fake_image/MatMul_grad/MatMulMatMulFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency Generator/fake_image/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ů
5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1MatMulGenerator/last_layer/leaky_reluFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ł
=gradients_1/Generator/fake_image/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Generator/fake_image/MatMul_grad/MatMul6^gradients_1/Generator/fake_image/MatMul_grad/MatMul_1
Á
Egradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/MatMul_grad/MatMul>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul
ż
Ggradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*H
_class>
<:loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

6gradients_1/Generator/last_layer/leaky_relu_grad/ShapeShape#Generator/last_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
°
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
˝
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2ShapeEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ű
6gradients_1/Generator/last_layer/leaky_relu_grad/zerosFill8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Fgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Generator/last_layer/leaky_relu_grad/Shape8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
˛
7gradients_1/Generator/last_layer/leaky_relu_grad/SelectSelect=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency6gradients_1/Generator/last_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Select=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqual6gradients_1/Generator/last_layer/leaky_relu_grad/zerosEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ü
4gradients_1/Generator/last_layer/leaky_relu_grad/SumSum7gradients_1/Generator/last_layer/leaky_relu_grad/SelectFgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ň
8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Generator/last_layer/leaky_relu_grad/Sum6gradients_1/Generator/last_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Hgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ř
:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_18gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Á
Agradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape;^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1
Ó
Igradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeB^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
Kgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1B^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
´
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

Jgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
÷
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulMulIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

8gradients_1/Generator/last_layer/leaky_relu/mul_grad/SumSum8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulJgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ě
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
ć
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Mul%Generator/last_layer/leaky_relu/alphaIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
Í
Egradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
Ń
Mgradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*O
_classE
CAloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape
é
Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*Q
_classG
ECloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
Ă
gradients_1/AddN_3AddNKgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Generator/last_layer/batch_normalization/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0

Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
Ů
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_3_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˝
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_3agradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ś
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
ˇ
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
ť
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Generator/last_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:

Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ů
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ł
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˝
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Generator/last_layer/fully_connected/BiasAddbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ś
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
ˇ
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
°
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
Ţ
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg
ť
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1

bgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:
ů
Igradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Ngradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad
°
Vgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Xgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*\
_classR
PNloc:@gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0

Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Generator/last_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:*
T0

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Generator/last_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
˘
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
¨
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
°
Cgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Generator/last_layer/fully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0

Egradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/third_layer/leaky_reluVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ă
Mgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1

Ugradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*V
_classL
JHloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul
˙
Wgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

ý
gradients_1/AddN_4AddNdgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:*
T0
Á
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_43Generator/last_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
Č
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_48Generator/last_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:
ţ
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1

`gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:
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
˛
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
Î
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
ţ
7gradients_1/Generator/third_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
â
>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ggradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/third_layer/leaky_relu_grad/Shape9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ĺ
8gradients_1/Generator/third_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/third_layer/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/third_layer/leaky_relu_grad/zerosUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˙
5gradients_1/Generator/third_layer/leaky_relu_grad/SumSum8gradients_1/Generator/third_layer/leaky_relu_grad/SelectGgradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ő
9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/third_layer/leaky_relu_grad/Sum7gradients_1/Generator/third_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Igradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ű
;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_19gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ä
Bgradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
×
Jgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ý
Lgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
ś
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

Kgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ú
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

9gradients_1/Generator/third_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ď
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
é
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/third_layer/leaky_relu/alphaJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
:˙˙˙˙˙˙˙˙˙
Đ
Fgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
Ő
Ngradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
í
Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*R
_classH
FDloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
Ć
gradients_1/AddN_5AddNLgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape9Generator/third_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:

Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
Ü
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_5`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ŕ
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_5bgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
š
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
ť
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
egradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
˝
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape-Generator/third_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:

Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
Ü
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ś
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency7Generator/third_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumSumNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ŕ
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul-Generator/third_layer/fully_connected/BiasAddcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1bgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
š
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
ť
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
ŕ
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegNegegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpf^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1M^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg
ż
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
ű
Jgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradcgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Ogradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpd^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyK^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad
´
Wgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitycgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ygradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_17Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1:Generator/third_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulQ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Ś
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
Ź
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
ł
Dgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(

Fgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1MatMul!Generator/second_layer/leaky_reluWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ć
Ngradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1

Vgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*W
_classM
KIloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul

Xgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*Y
_classO
MKloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1

gradients_1/AddN_6AddNegradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
Ă
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_64Generator/third_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
Ę
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_69Generator/third_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:*
T0

Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpM^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1

agradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:
¤
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1

8gradients_1/Generator/second_layer/leaky_relu_grad/ShapeShape%Generator/second_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
´
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Đ
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2ShapeVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
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
:˙˙˙˙˙˙˙˙˙
ĺ
?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Hgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/Generator/second_layer/leaky_relu_grad/Shape:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
9gradients_1/Generator/second_layer/leaky_relu_grad/SelectSelect?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency8gradients_1/Generator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Select?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqual8gradients_1/Generator/second_layer/leaky_relu_grad/zerosVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

6gradients_1/Generator/second_layer/leaky_relu_grad/SumSum9gradients_1/Generator/second_layer/leaky_relu_grad/SelectHgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ř
:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeReshape6gradients_1/Generator/second_layer/leaky_relu_grad/Sum8gradients_1/Generator/second_layer/leaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1Sum;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Jgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ţ
<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1Reshape8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
Cgradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_depsNoOp;^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape=^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1
Ű
Kgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeD^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*M
_classC
A?loc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape
á
Mgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1D^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
¸
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
 
Lgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ý
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulMulKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

:gradients_1/Generator/second_layer/leaky_relu/mul_grad/SumSum:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulLgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ň
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeReshape:gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ě
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Mul'Generator/second_layer/leaky_relu/alphaKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Ngradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ó
Ggradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp?^gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeA^gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
Ů
Ogradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeH^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*Q
_classG
ECloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape
ń
Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1H^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*S
_classI
GEloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
É
gradients_1/AddN_7AddNMgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
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
ß
agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_7agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ă
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_7cgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ź
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
ż
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
ż
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
ß
agradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Š
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency8Generator/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mulagradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ă
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul.Generator/second_layer/fully_connected/BiasAdddgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1cgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ź
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
ż
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:*
T0
â
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/NegNegfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:*
T0

Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpg^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1N^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
Ă
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
¤
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:
ý
Kgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGraddgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Pgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpe^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyL^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
¸
Xgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitydgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Zgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*^
_classT
RPloc:@gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad

Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_18Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
Ą
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Muldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1;Generator/second_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulR^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Ş
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
°
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
ś
Egradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulMatMulXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency2Generator/second_layer/fully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0

Ggradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/first_layer/leaky_reluXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
é
Ogradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpF^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulH^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1

Wgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityEgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulP^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ygradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityGgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1P^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:


gradients_1/AddN_8AddNfgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
Ĺ
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_85Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
Ě
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_8:Generator/second_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:*
T0

Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpN^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
˘
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:
¨
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:

7gradients_1/Generator/first_layer/leaky_relu_grad/ShapeShape$Generator/first_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
Ś
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Đ
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2ShapeWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0

=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ţ
7gradients_1/Generator/first_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ö
>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ggradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/first_layer/leaky_relu_grad/Shape9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ç
8gradients_1/Generator/first_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
É
:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/first_layer/leaky_relu_grad/zerosWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
5gradients_1/Generator/first_layer/leaky_relu_grad/SumSum8gradients_1/Generator/first_layer/leaky_relu_grad/SelectGgradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ő
9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/first_layer/leaky_relu_grad/Sum7gradients_1/Generator/first_layer/leaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Igradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ű
;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_19gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Bgradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
×
Jgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*L
_classB
@>loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape
Ý
Lgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ş
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Kgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
î
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_1/Generator/first_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ď
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
é
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/first_layer/leaky_relu/alphaJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Đ
Fgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
Ő
Ngradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
í
Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
gradients_1/AddN_9AddNLgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
Jgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
data_formatNHWC*
_output_shapes	
:*
T0
š
Ogradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_9K^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ě
Wgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_9P^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ygradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
˛
Dgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/first_layer/fully_connected/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d*
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
ć
Ngradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1

Vgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*W
_classM
KIloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d*
T0
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
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@Generator/fake_image/bias
Â
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
 *wž?*
dtype0

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
Â
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias
|
beta2_power_1/readIdentitybeta2_power_1*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: *
T0
ĺ
SGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
Ď
IGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
č
CGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d
č
1Generator/first_layer/fully_connected/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d
Î
8Generator/first_layer/fully_connected/kernel/Adam/AssignAssign1Generator/first_layer/fully_connected/kernel/AdamCGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0
ŕ
6Generator/first_layer/fully_connected/kernel/Adam/readIdentity1Generator/first_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
ç
UGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
Ń
KGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
î
EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d
ę
3Generator/first_layer/fully_connected/kernel/Adam_1
VariableV2*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name 
Ô
:Generator/first_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/first_layer/fully_connected/kernel/Adam_1EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
ä
8Generator/first_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/first_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
Ď
AGenerator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ü
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
Â
6Generator/first_layer/fully_connected/bias/Adam/AssignAssign/Generator/first_layer/fully_connected/bias/AdamAGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ö
4Generator/first_layer/fully_connected/bias/Adam/readIdentity/Generator/first_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:
Ń
CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ţ
1Generator/first_layer/fully_connected/bias/Adam_1
VariableV2*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Č
8Generator/first_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/first_layer/fully_connected/bias/Adam_1CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ú
6Generator/first_layer/fully_connected/bias/Adam_1/readIdentity1Generator/first_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:
ç
TGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ń
JGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
í
DGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillTGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorJGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ě
2Generator/second_layer/fully_connected/kernel/Adam
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
Ó
9Generator/second_layer/fully_connected/kernel/Adam/AssignAssign2Generator/second_layer/fully_connected/kernel/AdamDGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ä
7Generator/second_layer/fully_connected/kernel/Adam/readIdentity2Generator/second_layer/fully_connected/kernel/Adam* 
_output_shapes
:
*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
é
VGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ó
LGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ó
FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillVGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0
î
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
Ů
;Generator/second_layer/fully_connected/kernel/Adam_1/AssignAssign4Generator/second_layer/fully_connected/kernel/Adam_1FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(
č
9Generator/second_layer/fully_connected/kernel/Adam_1/readIdentity4Generator/second_layer/fully_connected/kernel/Adam_1*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
Ń
BGenerator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ţ
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
Ć
7Generator/second_layer/fully_connected/bias/Adam/AssignAssign0Generator/second_layer/fully_connected/bias/AdamBGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ů
5Generator/second_layer/fully_connected/bias/Adam/readIdentity0Generator/second_layer/fully_connected/bias/Adam*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:
Ó
DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ŕ
2Generator/second_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:
Ě
9Generator/second_layer/fully_connected/bias/Adam_1/AssignAssign2Generator/second_layer/fully_connected/bias/Adam_1DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ý
7Generator/second_layer/fully_connected/bias/Adam_1/readIdentity2Generator/second_layer/fully_connected/bias/Adam_1*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:
Ű
GGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
č
5Generator/second_layer/batch_normalization/gamma/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
Ú
<Generator/second_layer/batch_normalization/gamma/Adam/AssignAssign5Generator/second_layer/batch_normalization/gamma/AdamGGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
č
:Generator/second_layer/batch_normalization/gamma/Adam/readIdentity5Generator/second_layer/batch_normalization/gamma/Adam*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:
Ý
IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*
_output_shapes	
:*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*    *
dtype0
ę
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
ŕ
>Generator/second_layer/batch_normalization/gamma/Adam_1/AssignAssign7Generator/second_layer/batch_normalization/gamma/Adam_1IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
ě
<Generator/second_layer/batch_normalization/gamma/Adam_1/readIdentity7Generator/second_layer/batch_normalization/gamma/Adam_1*
_output_shapes	
:*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
Ů
FGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ć
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
Ö
;Generator/second_layer/batch_normalization/beta/Adam/AssignAssign4Generator/second_layer/batch_normalization/beta/AdamFGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
ĺ
9Generator/second_layer/batch_normalization/beta/Adam/readIdentity4Generator/second_layer/batch_normalization/beta/Adam*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:
Ű
HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0
č
6Generator/second_layer/batch_normalization/beta/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:
Ü
=Generator/second_layer/batch_normalization/beta/Adam_1/AssignAssign6Generator/second_layer/batch_normalization/beta/Adam_1HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
é
;Generator/second_layer/batch_normalization/beta/Adam_1/readIdentity6Generator/second_layer/batch_normalization/beta/Adam_1*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:
ĺ
SGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0
Ď
IGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
é
CGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ę
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
Ď
8Generator/third_layer/fully_connected/kernel/Adam/AssignAssign1Generator/third_layer/fully_connected/kernel/AdamCGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

á
6Generator/third_layer/fully_connected/kernel/Adam/readIdentity1Generator/third_layer/fully_connected/kernel/Adam* 
_output_shapes
:
*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
ç
UGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ń
KGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ď
EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0
ě
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
Ő
:Generator/third_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/third_layer/fully_connected/kernel/Adam_1EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ĺ
8Generator/third_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/third_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

Ď
AGenerator/third_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ü
/Generator/third_layer/fully_connected/bias/Adam
VariableV2*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Â
6Generator/third_layer/fully_connected/bias/Adam/AssignAssign/Generator/third_layer/fully_connected/bias/AdamAGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ö
4Generator/third_layer/fully_connected/bias/Adam/readIdentity/Generator/third_layer/fully_connected/bias/Adam*
_output_shapes	
:*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
Ń
CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ţ
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
Č
8Generator/third_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/third_layer/fully_connected/bias/Adam_1CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ú
6Generator/third_layer/fully_connected/bias/Adam_1/readIdentity1Generator/third_layer/fully_connected/bias/Adam_1*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:*
T0
Ů
FGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ć
4Generator/third_layer/batch_normalization/gamma/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma
Ö
;Generator/third_layer/batch_normalization/gamma/Adam/AssignAssign4Generator/third_layer/batch_normalization/gamma/AdamFGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
ĺ
9Generator/third_layer/batch_normalization/gamma/Adam/readIdentity4Generator/third_layer/batch_normalization/gamma/Adam*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
Ű
HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*    
č
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
Ü
=Generator/third_layer/batch_normalization/gamma/Adam_1/AssignAssign6Generator/third_layer/batch_normalization/gamma/Adam_1HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
é
;Generator/third_layer/batch_normalization/gamma/Adam_1/readIdentity6Generator/third_layer/batch_normalization/gamma/Adam_1*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
×
EGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ä
3Generator/third_layer/batch_normalization/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:
Ň
:Generator/third_layer/batch_normalization/beta/Adam/AssignAssign3Generator/third_layer/batch_normalization/beta/AdamEGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
â
8Generator/third_layer/batch_normalization/beta/Adam/readIdentity3Generator/third_layer/batch_normalization/beta/Adam*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:*
T0
Ů
GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    
ć
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
Ř
<Generator/third_layer/batch_normalization/beta/Adam_1/AssignAssign5Generator/third_layer/batch_normalization/beta/Adam_1GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ć
:Generator/third_layer/batch_normalization/beta/Adam_1/readIdentity5Generator/third_layer/batch_normalization/beta/Adam_1*
_output_shapes	
:*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
ă
RGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Í
HGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ĺ
BGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zerosFillRGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

č
0Generator/last_layer/fully_connected/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container 
Ë
7Generator/last_layer/fully_connected/kernel/Adam/AssignAssign0Generator/last_layer/fully_connected/kernel/AdamBGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

Ţ
5Generator/last_layer/fully_connected/kernel/Adam/readIdentity0Generator/last_layer/fully_connected/kernel/Adam*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
ĺ
TGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ď
JGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ë
DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillTGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorJGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ę
2Generator/last_layer/fully_connected/kernel/Adam_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
Ń
9Generator/last_layer/fully_connected/kernel/Adam_1/AssignAssign2Generator/last_layer/fully_connected/kernel/Adam_1DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
â
7Generator/last_layer/fully_connected/kernel/Adam_1/readIdentity2Generator/last_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
Ů
PGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
É
FGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Ř
@Generator/last_layer/fully_connected/bias/Adam/Initializer/zerosFillPGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorFGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:
Ú
.Generator/last_layer/fully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:
ž
5Generator/last_layer/fully_connected/bias/Adam/AssignAssign.Generator/last_layer/fully_connected/bias/Adam@Generator/last_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ó
3Generator/last_layer/fully_connected/bias/Adam/readIdentity.Generator/last_layer/fully_connected/bias/Adam*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:
Ű
RGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
Ë
HGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Ţ
BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zerosFillRGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0
Ü
0Generator/last_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:
Ä
7Generator/last_layer/fully_connected/bias/Adam_1/AssignAssign0Generator/last_layer/fully_connected/bias/Adam_1BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
×
5Generator/last_layer/fully_connected/bias/Adam_1/readIdentity0Generator/last_layer/fully_connected/bias/Adam_1*
_output_shapes	
:*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
ă
UGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0*
_output_shapes
:
Ó
KGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ě
EGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zerosFillUGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorKGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:
ä
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
Ň
:Generator/last_layer/batch_normalization/gamma/Adam/AssignAssign3Generator/last_layer/batch_normalization/gamma/AdamEGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma
â
8Generator/last_layer/batch_normalization/gamma/Adam/readIdentity3Generator/last_layer/batch_normalization/gamma/Adam*
_output_shapes	
:*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma
ĺ
WGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0*
_output_shapes
:
Ő
MGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ň
GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zerosFillWGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorMGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:
ć
5Generator/last_layer/batch_normalization/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:
Ř
<Generator/last_layer/batch_normalization/gamma/Adam_1/AssignAssign5Generator/last_layer/batch_normalization/gamma/Adam_1GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
ć
:Generator/last_layer/batch_normalization/gamma/Adam_1/readIdentity5Generator/last_layer/batch_normalization/gamma/Adam_1*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:
á
TGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:
Ń
JGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
č
DGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zerosFillTGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorJGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:
â
2Generator/last_layer/batch_normalization/beta/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container 
Î
9Generator/last_layer/batch_normalization/beta/Adam/AssignAssign2Generator/last_layer/batch_normalization/beta/AdamDGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
ß
7Generator/last_layer/batch_normalization/beta/Adam/readIdentity2Generator/last_layer/batch_normalization/beta/Adam*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:
ă
VGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:*
dtype0*
_output_shapes
:
Ó
LGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
î
FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zerosFillVGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:
ä
4Generator/last_layer/batch_normalization/beta/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:*
dtype0
Ô
;Generator/last_layer/batch_normalization/beta/Adam_1/AssignAssign4Generator/last_layer/batch_normalization/beta/Adam_1FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ă
9Generator/last_layer/batch_normalization/beta/Adam_1/readIdentity4Generator/last_layer/batch_normalization/beta/Adam_1*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:
Ă
BGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
­
8Generator/fake_image/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0
Ľ
2Generator/fake_image/kernel/Adam/Initializer/zerosFillBGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensor8Generator/fake_image/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0
Č
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
'Generator/fake_image/kernel/Adam/AssignAssign Generator/fake_image/kernel/Adam2Generator/fake_image/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:

Ž
%Generator/fake_image/kernel/Adam/readIdentity Generator/fake_image/kernel/Adam*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:

Ĺ
DGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ż
:Generator/fake_image/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ť
4Generator/fake_image/kernel/Adam_1/Initializer/zerosFillDGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensor:Generator/fake_image/kernel/Adam_1/Initializer/zeros/Const*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:

Ę
"Generator/fake_image/kernel/Adam_1
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container 

)Generator/fake_image/kernel/Adam_1/AssignAssign"Generator/fake_image/kernel/Adam_14Generator/fake_image/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:

˛
'Generator/fake_image/kernel/Adam_1/readIdentity"Generator/fake_image/kernel/Adam_1*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:

­
0Generator/fake_image/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    
ş
Generator/fake_image/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:
ţ
%Generator/fake_image/bias/Adam/AssignAssignGenerator/fake_image/bias/Adam0Generator/fake_image/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
Ł
#Generator/fake_image/bias/Adam/readIdentityGenerator/fake_image/bias/Adam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
Ż
2Generator/fake_image/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0*
_output_shapes	
:
ź
 Generator/fake_image/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:*
dtype0

'Generator/fake_image/bias/Adam_1/AssignAssign Generator/fake_image/bias/Adam_12Generator/fake_image/bias/Adam_1/Initializer/zeros*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
§
%Generator/fake_image/bias/Adam_1/readIdentity Generator/fake_image/bias/Adam_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
Y
Adam_1/learning_rateConst*
valueB
 *ˇQ9*
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
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
˝
DAdam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/first_layer/fully_connected/kernel1Generator/first_layer/fully_connected/kernel/Adam3Generator/first_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	d*
use_locking( *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
use_nesterov( 
°
BAdam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/first_layer/fully_connected/bias/Generator/first_layer/fully_connected/bias/Adam1Generator/first_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:
Ä
EAdam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam-Generator/second_layer/fully_connected/kernel2Generator/second_layer/fully_connected/kernel/Adam4Generator/second_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
ś
CAdam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam+Generator/second_layer/fully_connected/bias0Generator/second_layer/fully_connected/bias/Adam2Generator/second_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
Ů
HAdam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam0Generator/second_layer/batch_normalization/gamma5Generator/second_layer/batch_normalization/gamma/Adam7Generator/second_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
use_nesterov( 
Ň
GAdam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam/Generator/second_layer/batch_normalization/beta4Generator/second_layer/batch_normalization/beta/Adam6Generator/second_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:*
use_locking( 
ž
DAdam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/third_layer/fully_connected/kernel1Generator/third_layer/fully_connected/kernel/Adam3Generator/third_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
*
use_locking( *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
use_nesterov( 
°
BAdam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/third_layer/fully_connected/bias/Generator/third_layer/fully_connected/bias/Adam1Generator/third_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Ó
GAdam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam/Generator/third_layer/batch_normalization/gamma4Generator/third_layer/batch_normalization/gamma/Adam6Generator/third_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:
Ě
FAdam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdam	ApplyAdam.Generator/third_layer/batch_normalization/beta3Generator/third_layer/batch_normalization/beta/Adam5Generator/third_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:
¸
CAdam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Generator/last_layer/fully_connected/kernel0Generator/last_layer/fully_connected/kernel/Adam2Generator/last_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
Ş
AAdam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Generator/last_layer/fully_connected/bias.Generator/last_layer/fully_connected/bias/Adam0Generator/last_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
Í
FAdam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Generator/last_layer/batch_normalization/gamma3Generator/last_layer/batch_normalization/gamma/Adam5Generator/last_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:
Ć
EAdam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Generator/last_layer/batch_normalization/beta2Generator/last_layer/batch_normalization/beta/Adam4Generator/last_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
Ř
3Adam_1/update_Generator/fake_image/kernel/ApplyAdam	ApplyAdamGenerator/fake_image/kernel Generator/fake_image/kernel/Adam"Generator/fake_image/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
*
use_locking( *
T0*.
_class$
" loc:@Generator/fake_image/kernel*
use_nesterov( 
Ę
1Adam_1/update_Generator/fake_image/bias/ApplyAdam	ApplyAdamGenerator/fake_image/biasGenerator/fake_image/bias/Adam Generator/fake_image/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Ő	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*,
_class"
 loc:@Generator/fake_image/bias
Ş
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias
×	
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta22^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: *
T0
Ž
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: "yăÖl     ´C	ÇçÉęýÖAJÉŮ
ß˝
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
î
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
Ttype*1.12.02v1.12.0-0-ga6d8ffae09ö
u
Generator/noise_inPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d*
shape:˙˙˙˙˙˙˙˙˙d
ß
MGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
Ń
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&ž*
dtype0*
_output_shapes
: 
Ń
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&>*
dtype0*
_output_shapes
: 
Ć
UGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*

seed *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
seed2 *
dtype0*
_output_shapes
:	d
Î
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
: 
á
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
Ó
GGenerator/first_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
:	d*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
ă
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
Č
3Generator/first_layer/fully_connected/kernel/AssignAssign,Generator/first_layer/fully_connected/kernelGGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d
Ö
1Generator/first_layer/fully_connected/kernel/readIdentity,Generator/first_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
Ę
<Generator/first_layer/fully_connected/bias/Initializer/zerosConst*
_output_shapes	
:*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    *
dtype0
×
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
ł
1Generator/first_layer/fully_connected/bias/AssignAssign*Generator/first_layer/fully_connected/bias<Generator/first_layer/fully_connected/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(
Ě
/Generator/first_layer/fully_connected/bias/readIdentity*Generator/first_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:
Ö
,Generator/first_layer/fully_connected/MatMulMatMulGenerator/noise_in1Generator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
á
-Generator/first_layer/fully_connected/BiasAddBiasAdd,Generator/first_layer/fully_connected/MatMul/Generator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
k
&Generator/first_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
ľ
$Generator/first_layer/leaky_relu/mulMul&Generator/first_layer/leaky_relu/alpha-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
 Generator/first_layer/leaky_reluMaximum$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
NGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ó
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   ž
Ó
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   >
Ę
VGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformNGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ň
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
ć
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulVGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

Ř
HGenerator/second_layer/fully_connected/kernel/Initializer/random_uniformAddLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
ç
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
Í
4Generator/second_layer/fully_connected/kernel/AssignAssign-Generator/second_layer/fully_connected/kernelHGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

Ú
2Generator/second_layer/fully_connected/kernel/readIdentity-Generator/second_layer/fully_connected/kernel*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

Ě
=Generator/second_layer/fully_connected/bias/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ů
+Generator/second_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:
ˇ
2Generator/second_layer/fully_connected/bias/AssignAssign+Generator/second_layer/fully_connected/bias=Generator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ď
0Generator/second_layer/fully_connected/bias/readIdentity+Generator/second_layer/fully_connected/bias*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:
ć
-Generator/second_layer/fully_connected/MatMulMatMul Generator/first_layer/leaky_relu2Generator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ä
.Generator/second_layer/fully_connected/BiasAddBiasAdd-Generator/second_layer/fully_connected/MatMul0Generator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
AGenerator/second_layer/batch_normalization/gamma/Initializer/onesConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ă
0Generator/second_layer/batch_normalization/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container 
Ę
7Generator/second_layer/batch_normalization/gamma/AssignAssign0Generator/second_layer/batch_normalization/gammaAGenerator/second_layer/batch_normalization/gamma/Initializer/ones*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Ţ
5Generator/second_layer/batch_normalization/gamma/readIdentity0Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
Ô
AGenerator/second_layer/batch_normalization/beta/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0
á
/Generator/second_layer/batch_normalization/beta
VariableV2*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ç
6Generator/second_layer/batch_normalization/beta/AssignAssign/Generator/second_layer/batch_normalization/betaAGenerator/second_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
Ű
4Generator/second_layer/batch_normalization/beta/readIdentity/Generator/second_layer/batch_normalization/beta*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:
â
HGenerator/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ď
6Generator/second_layer/batch_normalization/moving_mean
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean
ă
=Generator/second_layer/batch_normalization/moving_mean/AssignAssign6Generator/second_layer/batch_normalization/moving_meanHGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:
đ
;Generator/second_layer/batch_normalization/moving_mean/readIdentity6Generator/second_layer/batch_normalization/moving_mean*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
_output_shapes	
:
é
KGenerator/second_layer/batch_normalization/moving_variance/Initializer/onesConst*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:
÷
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
ň
AGenerator/second_layer/batch_normalization/moving_variance/AssignAssign:Generator/second_layer/batch_normalization/moving_varianceKGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance
ü
?Generator/second_layer/batch_normalization/moving_variance/readIdentity:Generator/second_layer/batch_normalization/moving_variance*
_output_shapes	
:*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance

:Generator/second_layer/batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o:
â
8Generator/second_layer/batch_normalization/batchnorm/addAdd?Generator/second_layer/batch_normalization/moving_variance/read:Generator/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:
Ł
:Generator/second_layer/batch_normalization/batchnorm/RsqrtRsqrt8Generator/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
Ř
8Generator/second_layer/batch_normalization/batchnorm/mulMul:Generator/second_layer/batch_normalization/batchnorm/Rsqrt5Generator/second_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
Ţ
:Generator/second_layer/batch_normalization/batchnorm/mul_1Mul.Generator/second_layer/fully_connected/BiasAdd8Generator/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ţ
:Generator/second_layer/batch_normalization/batchnorm/mul_2Mul;Generator/second_layer/batch_normalization/moving_mean/read8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
×
8Generator/second_layer/batch_normalization/batchnorm/subSub4Generator/second_layer/batch_normalization/beta/read:Generator/second_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ę
:Generator/second_layer/batch_normalization/batchnorm/add_1Add:Generator/second_layer/batch_normalization/batchnorm/mul_18Generator/second_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
l
'Generator/second_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Ä
%Generator/second_layer/leaky_relu/mulMul'Generator/second_layer/leaky_relu/alpha:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
!Generator/second_layer/leaky_reluMaximum%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
MGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ń
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *óľ˝*
dtype0
Ń
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *óľ=*
dtype0*
_output_shapes
: 
Ç
UGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*

seed *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
seed2 *
dtype0
Î
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
_output_shapes
: 
â
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

Ô
GGenerator/third_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:

ĺ
,Generator/third_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:

É
3Generator/third_layer/fully_connected/kernel/AssignAssign,Generator/third_layer/fully_connected/kernelGGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(
×
1Generator/third_layer/fully_connected/kernel/readIdentity,Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
Ę
<Generator/third_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
×
*Generator/third_layer/fully_connected/bias
VariableV2*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ł
1Generator/third_layer/fully_connected/bias/AssignAssign*Generator/third_layer/fully_connected/bias<Generator/third_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ě
/Generator/third_layer/fully_connected/bias/readIdentity*Generator/third_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:
ĺ
,Generator/third_layer/fully_connected/MatMulMatMul!Generator/second_layer/leaky_relu1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
á
-Generator/third_layer/fully_connected/BiasAddBiasAdd,Generator/third_layer/fully_connected/MatMul/Generator/third_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ó
@Generator/third_layer/batch_normalization/gamma/Initializer/onesConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
á
/Generator/third_layer/batch_normalization/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:
Ć
6Generator/third_layer/batch_normalization/gamma/AssignAssign/Generator/third_layer/batch_normalization/gamma@Generator/third_layer/batch_normalization/gamma/Initializer/ones*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ű
4Generator/third_layer/batch_normalization/gamma/readIdentity/Generator/third_layer/batch_normalization/gamma*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
Ň
@Generator/third_layer/batch_normalization/beta/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ß
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
Ă
5Generator/third_layer/batch_normalization/beta/AssignAssign.Generator/third_layer/batch_normalization/beta@Generator/third_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
Ř
3Generator/third_layer/batch_normalization/beta/readIdentity.Generator/third_layer/batch_normalization/beta*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:
ŕ
GGenerator/third_layer/batch_normalization/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
valueB*    
í
5Generator/third_layer/batch_normalization/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
	container 
ß
<Generator/third_layer/batch_normalization/moving_mean/AssignAssign5Generator/third_layer/batch_normalization/moving_meanGGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:
í
:Generator/third_layer/batch_normalization/moving_mean/readIdentity5Generator/third_layer/batch_normalization/moving_mean*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
_output_shapes	
:*
T0
ç
JGenerator/third_layer/batch_normalization/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
valueB*  ?
ő
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
î
@Generator/third_layer/batch_normalization/moving_variance/AssignAssign9Generator/third_layer/batch_normalization/moving_varianceJGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:
ů
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
ß
7Generator/third_layer/batch_normalization/batchnorm/addAdd>Generator/third_layer/batch_normalization/moving_variance/read9Generator/third_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:
Ą
9Generator/third_layer/batch_normalization/batchnorm/RsqrtRsqrt7Generator/third_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
Ő
7Generator/third_layer/batch_normalization/batchnorm/mulMul9Generator/third_layer/batch_normalization/batchnorm/Rsqrt4Generator/third_layer/batch_normalization/gamma/read*
_output_shapes	
:*
T0
Ű
9Generator/third_layer/batch_normalization/batchnorm/mul_1Mul-Generator/third_layer/fully_connected/BiasAdd7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
9Generator/third_layer/batch_normalization/batchnorm/mul_2Mul:Generator/third_layer/batch_normalization/moving_mean/read7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:
Ô
7Generator/third_layer/batch_normalization/batchnorm/subSub3Generator/third_layer/batch_normalization/beta/read9Generator/third_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:*
T0
ç
9Generator/third_layer/batch_normalization/batchnorm/add_1Add9Generator/third_layer/batch_normalization/batchnorm/mul_17Generator/third_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
k
&Generator/third_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Á
$Generator/third_layer/leaky_relu/mulMul&Generator/third_layer/leaky_relu/alpha9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ż
 Generator/third_layer/leaky_reluMaximum$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
LGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0
Ď
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  ˝*
dtype0*
_output_shapes
: 
Ď
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
Ä
TGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*

seed *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
seed2 *
dtype0
Ę
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/subSubJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
_output_shapes
: 
Ţ
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
Đ
FGenerator/last_layer/fully_connected/kernel/Initializer/random_uniformAddJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

ă
+Generator/last_layer/fully_connected/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container 
Ĺ
2Generator/last_layer/fully_connected/kernel/AssignAssign+Generator/last_layer/fully_connected/kernelFGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ô
0Generator/last_layer/fully_connected/kernel/readIdentity+Generator/last_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

Ô
KGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0
Ä
AGenerator/last_layer/fully_connected/bias/Initializer/zeros/ConstConst*
_output_shapes
: *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0
É
;Generator/last_layer/fully_connected/bias/Initializer/zerosFillKGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorAGenerator/last_layer/fully_connected/bias/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:
Ő
)Generator/last_layer/fully_connected/bias
VariableV2*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ż
0Generator/last_layer/fully_connected/bias/AssignAssign)Generator/last_layer/fully_connected/bias;Generator/last_layer/fully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
É
.Generator/last_layer/fully_connected/bias/readIdentity)Generator/last_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:
â
+Generator/last_layer/fully_connected/MatMulMatMul Generator/third_layer/leaky_relu0Generator/last_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ţ
,Generator/last_layer/fully_connected/BiasAddBiasAdd+Generator/last_layer/fully_connected/MatMul.Generator/last_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
OGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0*
_output_shapes
:
Í
EGenerator/last_layer/batch_normalization/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *  ?
Ú
?Generator/last_layer/batch_normalization/gamma/Initializer/onesFillOGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorEGenerator/last_layer/batch_normalization/gamma/Initializer/ones/Const*
_output_shapes	
:*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
ß
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
Â
5Generator/last_layer/batch_normalization/gamma/AssignAssign.Generator/last_layer/batch_normalization/gamma?Generator/last_layer/batch_normalization/gamma/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(
Ř
3Generator/last_layer/batch_normalization/gamma/readIdentity.Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma
Ü
OGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:*
dtype0*
_output_shapes
:
Ě
EGenerator/last_layer/batch_normalization/beta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    
Ů
?Generator/last_layer/batch_normalization/beta/Initializer/zerosFillOGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorEGenerator/last_layer/batch_normalization/beta/Initializer/zeros/Const*
_output_shapes	
:*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0
Ý
-Generator/last_layer/batch_normalization/beta
VariableV2*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ż
4Generator/last_layer/batch_normalization/beta/AssignAssign-Generator/last_layer/batch_normalization/beta?Generator/last_layer/batch_normalization/beta/Initializer/zeros*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ő
2Generator/last_layer/batch_normalization/beta/readIdentity-Generator/last_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:
ę
VGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB:*
dtype0*
_output_shapes
:
Ú
LGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
ő
FGenerator/last_layer/batch_normalization/moving_mean/Initializer/zerosFillVGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/Const*
_output_shapes	
:*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*

index_type0
ë
4Generator/last_layer/batch_normalization/moving_mean
VariableV2*
shared_name *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
Ű
;Generator/last_layer/batch_normalization/moving_mean/AssignAssign4Generator/last_layer/batch_normalization/moving_meanFGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
validate_shape(
ę
9Generator/last_layer/batch_normalization/moving_mean/readIdentity4Generator/last_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
_output_shapes	
:
ń
YGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB:*
dtype0*
_output_shapes
:
á
OGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/ConstConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 

IGenerator/last_layer/batch_normalization/moving_variance/Initializer/onesFillYGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorOGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/Const*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*

index_type0*
_output_shapes	
:*
T0
ó
8Generator/last_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
	container *
shape:
ę
?Generator/last_layer/batch_normalization/moving_variance/AssignAssign8Generator/last_layer/batch_normalization/moving_varianceIGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:
ö
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
Ü
6Generator/last_layer/batch_normalization/batchnorm/addAdd=Generator/last_layer/batch_normalization/moving_variance/read8Generator/last_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:*
T0

8Generator/last_layer/batch_normalization/batchnorm/RsqrtRsqrt6Generator/last_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:
Ň
6Generator/last_layer/batch_normalization/batchnorm/mulMul8Generator/last_layer/batch_normalization/batchnorm/Rsqrt3Generator/last_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
Ř
8Generator/last_layer/batch_normalization/batchnorm/mul_1Mul,Generator/last_layer/fully_connected/BiasAdd6Generator/last_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ř
8Generator/last_layer/batch_normalization/batchnorm/mul_2Mul9Generator/last_layer/batch_normalization/moving_mean/read6Generator/last_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:*
T0
Ń
6Generator/last_layer/batch_normalization/batchnorm/subSub2Generator/last_layer/batch_normalization/beta/read8Generator/last_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:
ä
8Generator/last_layer/batch_normalization/batchnorm/add_1Add8Generator/last_layer/batch_normalization/batchnorm/mul_16Generator/last_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
%Generator/last_layer/leaky_relu/alphaConst*
_output_shapes
: *
valueB
 *ÍĚL>*
dtype0
ž
#Generator/last_layer/leaky_relu/mulMul%Generator/last_layer/leaky_relu/alpha8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
Generator/last_layer/leaky_reluMaximum#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
<Generator/fake_image/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ż
:Generator/fake_image/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zők˝*
dtype0*
_output_shapes
: 
Ż
:Generator/fake_image/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *zők=*
dtype0*
_output_shapes
: 

DGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniformRandomUniform<Generator/fake_image/kernel/Initializer/random_uniform/shape*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
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
6Generator/fake_image/kernel/Initializer/random_uniformAdd:Generator/fake_image/kernel/Initializer/random_uniform/mul:Generator/fake_image/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:

Ă
Generator/fake_image/kernel
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
¤
 Generator/fake_image/kernel/readIdentityGenerator/fake_image/kernel*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:

¨
+Generator/fake_image/bias/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0*
_output_shapes	
:
ľ
Generator/fake_image/bias
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ď
 Generator/fake_image/bias/AssignAssignGenerator/fake_image/bias+Generator/fake_image/bias/Initializer/zeros*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

Generator/fake_image/bias/readIdentityGenerator/fake_image/bias*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
Á
Generator/fake_image/MatMulMatMulGenerator/last_layer/leaky_relu Generator/fake_image/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
Ž
Generator/fake_image/BiasAddBiasAddGenerator/fake_image/MatMulGenerator/fake_image/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
r
Generator/fake_image/TanhTanhGenerator/fake_image/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
Discriminator/real_inPlaceholder*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
ç
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ů
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY˝*
dtype0*
_output_shapes
: 
Ů
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
Ó
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
Ţ
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
_output_shapes
: 
ň
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:

ä
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
í
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
Ů
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

ă
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
Ň
@Discriminator/first_layer/fully_connected/bias/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ß
.Discriminator/first_layer/fully_connected/bias
VariableV2*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ă
5Discriminator/first_layer/fully_connected/bias/AssignAssign.Discriminator/first_layer/fully_connected/bias@Discriminator/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ř
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:
á
0Discriminator/first_layer/fully_connected/MatMulMatMulDiscriminator/real_in5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
í
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
o
*Discriminator/first_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Á
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ż
$Discriminator/first_layer/leaky_reluMaximum(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
RDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ű
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *óľ˝*
dtype0*
_output_shapes
: 
Ű
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *óľ=*
dtype0*
_output_shapes
: 
Ö
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
â
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
_output_shapes
: 
ö
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

č
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
ď
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
Ý
8Discriminator/second_layer/fully_connected/kernel/AssignAssign1Discriminator/second_layer/fully_connected/kernelLDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

ć
6Discriminator/second_layer/fully_connected/kernel/readIdentity1Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
Ô
ADiscriminator/second_layer/fully_connected/bias/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
á
/Discriminator/second_layer/fully_connected/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
Ç
6Discriminator/second_layer/fully_connected/bias/AssignAssign/Discriminator/second_layer/fully_connected/biasADiscriminator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ű
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
ň
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
đ
2Discriminator/second_layer/fully_connected/BiasAddBiasAdd1Discriminator/second_layer/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
+Discriminator/second_layer/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Ä
)Discriminator/second_layer/leaky_relu/mulMul+Discriminator/second_layer/leaky_relu/alpha2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
%Discriminator/second_layer/leaky_reluMaximum)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
:Discriminator/prob/kernel/Initializer/random_uniform/shapeConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ť
8Discriminator/prob/kernel/Initializer/random_uniform/minConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Ivž*
dtype0*
_output_shapes
: 
Ť
8Discriminator/prob/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@Discriminator/prob/kernel*
valueB
 *Iv>
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
8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
: *
T0
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
˝
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
ü
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
˘
)Discriminator/prob/bias/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
Ż
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
ć
Discriminator/prob/bias/AssignAssignDiscriminator/prob/bias)Discriminator/prob/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias

Discriminator/prob/bias/readIdentityDiscriminator/prob/bias*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
Â
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
§
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ç
2Discriminator/first_layer_1/fully_connected/MatMulMatMulGenerator/fake_image/Tanh5Discriminator/first_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ń
3Discriminator/first_layer_1/fully_connected/BiasAddBiasAdd2Discriminator/first_layer_1/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
,Discriminator/first_layer_1/leaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Ç
*Discriminator/first_layer_1/leaky_relu/mulMul,Discriminator/first_layer_1/leaky_relu/alpha3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ĺ
&Discriminator/first_layer_1/leaky_reluMaximum*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ö
3Discriminator/second_layer_1/fully_connected/MatMulMatMul&Discriminator/first_layer_1/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ô
4Discriminator/second_layer_1/fully_connected/BiasAddBiasAdd3Discriminator/second_layer_1/fully_connected/MatMul4Discriminator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
-Discriminator/second_layer_1/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL>
Ę
+Discriminator/second_layer_1/leaky_relu/mulMul-Discriminator/second_layer_1/leaky_relu/alpha4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
'Discriminator/second_layer_1/leaky_reluMaximum+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
Discriminator/prob_1/MatMulMatMul'Discriminator/second_layer_1/leaky_reluDiscriminator/prob/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
Ť
Discriminator/prob_1/BiasAddBiasAddDiscriminator/prob_1/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙*
T0
s
logistic_loss/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

logistic_loss/GreaterEqualGreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˘
logistic_loss/SelectSelectlogistic_loss/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
logistic_loss/NegNegDiscriminator/prob/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/NegDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
logistic_loss/mulMulDiscriminator/prob/BiasAdd	ones_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
logistic_loss/ExpExplogistic_loss/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙*
T0
w
logistic_loss_1/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

logistic_loss_1/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAdd
zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
logistic_loss_1/ExpExplogistic_loss_1/Select_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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
ones_like_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
}
ones_like_1Fillones_like_1/Shapeones_like_1/Const*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ş
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
logistic_loss_2/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
logistic_loss_2/mulMulDiscriminator/prob_1/BiasAddones_like_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
logistic_loss_2/ExpExplogistic_loss_2/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
X
Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
f
Mean_2Meanlogistic_loss_2Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
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
ą
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Fill$^gradients/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
ł
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
:˙˙˙˙˙˙˙˙˙
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
gradients/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0

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
:˙˙˙˙˙˙˙˙˙
t
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ł
gradients/Mean_1_grad/ReshapeReshape-gradients/add_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
j
gradients/Mean_1_grad/ShapeShapelogistic_loss_1*
T0*
out_type0*
_output_shapes
:
˘
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
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
:˙˙˙˙˙˙˙˙˙
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
Ň
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
¸
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ľ
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ź
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ť
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1

5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
Ř
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ž
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ť
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Á
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/logistic_loss_1_grad/tuple/group_depsNoOp'^gradients/logistic_loss_1_grad/Reshape)^gradients/logistic_loss_1_grad/Reshape_1

7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape

9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
out_type0*
_output_shapes
:*
T0
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
T0*
out_type0*
_output_shapes
:
Ţ
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ú
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Á
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ţ
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
Ĺ
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1

9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape

;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
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
:˙˙˙˙˙˙˙˙˙

-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
ä
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ŕ
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ç
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ä
(gradients/logistic_loss_1/sub_grad/Sum_1Sum7gradients/logistic_loss_1_grad/tuple/control_dependency:gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
z
&gradients/logistic_loss_1/sub_grad/NegNeg(gradients/logistic_loss_1/sub_grad/Sum_1*
T0*
_output_shapes
:
Ë
,gradients/logistic_loss_1/sub_grad/Reshape_1Reshape&gradients/logistic_loss_1/sub_grad/Neg*gradients/logistic_loss_1/sub_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

3gradients/logistic_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/sub_grad/Reshape-^gradients/logistic_loss_1/sub_grad/Reshape_1

;gradients/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/sub_grad/Reshape4^gradients/logistic_loss_1/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/sub_grad/Reshape
 
=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/sub_grad/Reshape_14^gradients/logistic_loss_1/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
*gradients/logistic_loss_1/Log1p_grad/add/xConst:^gradients/logistic_loss_1_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  ?
˘
(gradients/logistic_loss_1/Log1p_grad/addAdd*gradients/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

/gradients/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
í
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ď
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1

<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

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
Ţ
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
É
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Á
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
&gradients/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ç
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1

9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

0gradients/logistic_loss_1/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
,gradients/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual;gradients/logistic_loss_1/sub_grad/tuple/control_dependency0gradients/logistic_loss_1/Select_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
÷
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
¤
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select
Ş
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

(gradients/logistic_loss_1/mul_grad/ShapeShapeDiscriminator/prob_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
t
*gradients/logistic_loss_1/mul_grad/Shape_1Shape
zeros_like*
T0*
out_type0*
_output_shapes
:
ä
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ş
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1
zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
&gradients/logistic_loss_1/mul_grad/SumSum&gradients/logistic_loss_1/mul_grad/Mul8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ç
*gradients/logistic_loss_1/mul_grad/ReshapeReshape&gradients/logistic_loss_1/mul_grad/Sum(gradients/logistic_loss_1/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
(gradients/logistic_loss_1/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Í
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

3gradients/logistic_loss_1/mul_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/mul_grad/Reshape-^gradients/logistic_loss_1/mul_grad/Reshape_1

;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/mul_grad/Reshape4^gradients/logistic_loss_1/mul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/mul_grad/Reshape
 
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ţ
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
¤
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select
Ş
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
Ź
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select
˛
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
_output_shapes
:*
T0*
data_formatNHWC

:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN6^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad

Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
ł
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad
ý
gradients/AddN_1AddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
data_formatNHWC*
_output_shapes
:*
T0

<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_18^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad

Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
ö
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ö
1gradients/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
§
9gradients/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp0^gradients/Discriminator/prob/MatMul_grad/MatMul2^gradients/Discriminator/prob/MatMul_grad/MatMul_1
ą
Agradients/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity/gradients/Discriminator/prob/MatMul_grad/MatMul:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/Discriminator/prob/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ž
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1
ú
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ü
3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
­
;gradients/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp2^gradients/Discriminator/prob_1/MatMul_grad/MatMul4^gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
š
Cgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/Discriminator/prob_1/MatMul_grad/MatMul<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0

gradients/AddN_2AddNDgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
Ł
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ž
<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
˝
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
:˙˙˙˙˙˙˙˙˙
ă
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Jgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Discriminator/second_layer/leaky_relu_grad/Shape<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ş
;gradients/Discriminator/second_layer/leaky_relu_grad/SelectSelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency:gradients/Discriminator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

8gradients/Discriminator/second_layer/leaky_relu_grad/SumSum;gradients/Discriminator/second_layer/leaky_relu_grad/SelectJgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ţ
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
Í
Egradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp=^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape?^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
ă
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
é
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
§
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
˛
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Á
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
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
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
é
Cgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Â
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ä
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
ë
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape
ń
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
˛
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ś
Ngradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ů
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulMulMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ř
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ô
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ů
Igradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOpA^gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeC^gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1
á
Qgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeJ^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape
ů
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
ś
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ź
Pgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
˙
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ţ
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ú
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
ß
Kgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpC^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeE^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
é
Sgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeL^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape

Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1L^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
gradients/AddN_4AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N
Ť
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:
˝
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ó
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
Ó
gradients/AddN_5AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:
Á
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
Ů
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ž
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ś
Igradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_reluZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ď
Qgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpH^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulJ^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1

Ygradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityGgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulR^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

Â
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ź
Kgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ő
Sgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpJ^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulL^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1

[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulT^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*\
_classR
PNloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

ç
gradients/AddN_6AddN\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
Ą
9gradients/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ź
;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ô
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
:˙˙˙˙˙˙˙˙˙
ŕ
@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ď
:gradients/Discriminator/first_layer/leaky_relu_grad/SelectSelect@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency9gradients/Discriminator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ń
<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Select@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqual9gradients/Discriminator/first_layer/leaky_relu_grad/zerosYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

7gradients/Discriminator/first_layer/leaky_relu_grad/SumSum:gradients/Discriminator/first_layer/leaky_relu_grad/SelectIgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ű
;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape7gradients/Discriminator/first_layer/leaky_relu_grad/Sum9gradients/Discriminator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Kgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
Dgradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_depsNoOp<^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape>^gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1
ß
Lgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeE^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
°
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ř
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
;gradients/Discriminator/first_layer_1/leaky_relu_grad/zerosFill=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Agradients/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ć
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
×
<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ů
>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectBgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual;gradients/Discriminator/first_layer_1/leaky_relu_grad/zeros[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

9gradients/Discriminator/first_layer_1/leaky_relu_grad/SumSum<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectKgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
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
:˙˙˙˙˙˙˙˙˙
Đ
Fgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp>^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape@^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
ç
Ngradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeG^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
ć
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
°
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ł
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ö
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ő
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ń
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
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
:˙˙˙˙˙˙˙˙˙
Ö
Hgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp@^gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeB^gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
Ý
Pgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeI^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*R
_classH
FDloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
ő
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
´
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Š
Ogradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ü
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulOgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ű
Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
÷
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Qgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Cgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1Reshape?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ü
Jgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeD^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
ĺ
Rgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeK^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*T
_classJ
HFloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
ý
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
gradients/AddN_8AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:
ť
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Đ
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradR^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Đ
gradients/AddN_9AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
data_formatNHWC*
_output_shapes	
:*
T0
ż
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
Ö
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1

]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
ť
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(

Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/real_inYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ě
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1

Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul

Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

ż
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(

Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ň
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1

Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
ĺ
gradients/AddN_10AddN[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:
ä
gradients/AddN_11AddNZgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:

Ą
beta1_power/initial_valueConst*
_output_shapes
: *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *fff?*
dtype0
˛
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
Ń
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
Ą
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *wž?
˛
beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
Ń
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 

beta2_power/readIdentitybeta2_power*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: *
T0
í
WDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
×
MDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ů
GDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillWDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorMDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ň
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
ß
<Discriminator/first_layer/fully_connected/kernel/Adam/AssignAssign5Discriminator/first_layer/fully_connected/kernel/AdamGDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
í
:Discriminator/first_layer/fully_connected/kernel/Adam/readIdentity5Discriminator/first_layer/fully_connected/kernel/Adam*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:

ď
YDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ů
ODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
˙
IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillYDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0
ô
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
ĺ
>Discriminator/first_layer/fully_connected/kernel/Adam_1/AssignAssign7Discriminator/first_layer/fully_connected/kernel/Adam_1IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
ń
<Discriminator/first_layer/fully_connected/kernel/Adam_1/readIdentity7Discriminator/first_layer/fully_connected/kernel/Adam_1*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:

×
EDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ä
3Discriminator/first_layer/fully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:
Ň
:Discriminator/first_layer/fully_connected/bias/Adam/AssignAssign3Discriminator/first_layer/fully_connected/bias/AdamEDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(
â
8Discriminator/first_layer/fully_connected/bias/Adam/readIdentity3Discriminator/first_layer/fully_connected/bias/Adam*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:*
T0
Ů
GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB*    *
dtype0
ć
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
Ř
<Discriminator/first_layer/fully_connected/bias/Adam_1/AssignAssign5Discriminator/first_layer/fully_connected/bias/Adam_1GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
ć
:Discriminator/first_layer/fully_connected/bias/Adam_1/readIdentity5Discriminator/first_layer/fully_connected/bias/Adam_1*
_output_shapes	
:*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
ď
XDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ů
NDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    
ý
HDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillXDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorNDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ô
6Discriminator/second_layer/fully_connected/kernel/Adam
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
ă
=Discriminator/second_layer/fully_connected/kernel/Adam/AssignAssign6Discriminator/second_layer/fully_connected/kernel/AdamHDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
đ
;Discriminator/second_layer/fully_connected/kernel/Adam/readIdentity6Discriminator/second_layer/fully_connected/kernel/Adam*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:

ń
ZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ű
PDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorPDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ö
8Discriminator/second_layer/fully_connected/kernel/Adam_1
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
é
?Discriminator/second_layer/fully_connected/kernel/Adam_1/AssignAssign8Discriminator/second_layer/fully_connected/kernel/Adam_1JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ô
=Discriminator/second_layer/fully_connected/kernel/Adam_1/readIdentity8Discriminator/second_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
Ů
FDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0
ć
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
Ö
;Discriminator/second_layer/fully_connected/bias/Adam/AssignAssign4Discriminator/second_layer/fully_connected/bias/AdamFDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ĺ
9Discriminator/second_layer/fully_connected/bias/Adam/readIdentity4Discriminator/second_layer/fully_connected/bias/Adam*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:*
T0
Ű
HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
č
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
Ü
=Discriminator/second_layer/fully_connected/bias/Adam_1/AssignAssign6Discriminator/second_layer/fully_connected/bias/Adam_1HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
é
;Discriminator/second_layer/fully_connected/bias/Adam_1/readIdentity6Discriminator/second_layer/fully_connected/bias/Adam_1*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:
ľ
0Discriminator/prob/kernel/Adam/Initializer/zerosConst*
_output_shapes
:	*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0
Â
Discriminator/prob/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	

%Discriminator/prob/kernel/Adam/AssignAssignDiscriminator/prob/kernel/Adam0Discriminator/prob/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel
§
#Discriminator/prob/kernel/Adam/readIdentityDiscriminator/prob/kernel/Adam*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	
ˇ
2Discriminator/prob/kernel/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Ä
 Discriminator/prob/kernel/Adam_1
VariableV2*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel

'Discriminator/prob/kernel/Adam_1/AssignAssign Discriminator/prob/kernel/Adam_12Discriminator/prob/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	
Ť
%Discriminator/prob/kernel/Adam_1/readIdentity Discriminator/prob/kernel/Adam_1*
_output_shapes
:	*
T0*,
_class"
 loc:@Discriminator/prob/kernel
§
.Discriminator/prob/bias/Adam/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
´
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
ő
#Discriminator/prob/bias/Adam/AssignAssignDiscriminator/prob/bias/Adam.Discriminator/prob/bias/Adam/Initializer/zeros**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

!Discriminator/prob/bias/Adam/readIdentityDiscriminator/prob/bias/Adam*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
Š
0Discriminator/prob/bias/Adam_1/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
ś
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
ű
%Discriminator/prob/bias/Adam_1/AssignAssignDiscriminator/prob/bias/Adam_10Discriminator/prob/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(
 
#Discriminator/prob/bias/Adam_1/readIdentityDiscriminator/prob/bias/Adam_1*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *ˇQ9*
dtype0
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
 *wž?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
ý
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

î
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
Adam/beta2Adam/epsilongradients/AddN_7*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:

ň
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
Adam/beta2Adam/epsilongradients/AddN_3*
use_nesterov( *
_output_shapes
:	*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel
ů
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
Adam/beta1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
š
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
˝
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
Ž
AdamNoOp^Adam/Assign^Adam/Assign_1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam
T
gradients_1/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
gradients_1/Mean_2_grad/ShapeShapelogistic_loss_2*
out_type0*
_output_shapes
:*
T0
¨
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0
n
gradients_1/Mean_2_grad/Shape_1Shapelogistic_loss_2*
_output_shapes
:*
T0*
out_type0
b
gradients_1/Mean_2_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_1/Mean_2_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
˘
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_2_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Ś
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

gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0

 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 

gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0

gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
&gradients_1/logistic_loss_2_grad/ShapeShapelogistic_loss_2/sub*
_output_shapes
:*
T0*
out_type0
}
(gradients_1/logistic_loss_2_grad/Shape_1Shapelogistic_loss_2/Log1p*
out_type0*
_output_shapes
:*
T0
Ţ
6gradients_1/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_2_grad/Shape(gradients_1/logistic_loss_2_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ä
$gradients_1/logistic_loss_2_grad/SumSumgradients_1/Mean_2_grad/truediv6gradients_1/logistic_loss_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Á
(gradients_1/logistic_loss_2_grad/ReshapeReshape$gradients_1/logistic_loss_2_grad/Sum&gradients_1/logistic_loss_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Č
&gradients_1/logistic_loss_2_grad/Sum_1Sumgradients_1/Mean_2_grad/truediv8gradients_1/logistic_loss_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ç
*gradients_1/logistic_loss_2_grad/Reshape_1Reshape&gradients_1/logistic_loss_2_grad/Sum_1(gradients_1/logistic_loss_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

1gradients_1/logistic_loss_2_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_2_grad/Reshape+^gradients_1/logistic_loss_2_grad/Reshape_1

9gradients_1/logistic_loss_2_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_2_grad/Reshape2^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_2_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;gradients_1/logistic_loss_2_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_2_grad/Reshape_12^gradients_1/logistic_loss_2_grad/tuple/group_deps*=
_class3
1/loc:@gradients_1/logistic_loss_2_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

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
ę
:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/sub_grad/Shape,gradients_1/logistic_loss_2/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ć
(gradients_1/logistic_loss_2/sub_grad/SumSum9gradients_1/logistic_loss_2_grad/tuple/control_dependency:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Í
,gradients_1/logistic_loss_2/sub_grad/ReshapeReshape(gradients_1/logistic_loss_2/sub_grad/Sum*gradients_1/logistic_loss_2/sub_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ę
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
Ń
.gradients_1/logistic_loss_2/sub_grad/Reshape_1Reshape(gradients_1/logistic_loss_2/sub_grad/Neg,gradients_1/logistic_loss_2/sub_grad/Shape_1*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

5gradients_1/logistic_loss_2/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/sub_grad/Reshape/^gradients_1/logistic_loss_2/sub_grad/Reshape_1
˘
=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/sub_grad/Reshape6^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/sub_grad/Reshape
¨
?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/sub_grad/Reshape_16^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
,gradients_1/logistic_loss_2/Log1p_grad/add/xConst<^gradients_1/logistic_loss_2_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ś
*gradients_1/logistic_loss_2/Log1p_grad/addAdd,gradients_1/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

1gradients_1/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_2/Log1p_grad/add*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
*gradients_1/logistic_loss_2/Log1p_grad/mulMul;gradients_1/logistic_loss_2_grad/tuple/control_dependency_11gradients_1/logistic_loss_2/Log1p_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

2gradients_1/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ű
.gradients_1/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_2/Select_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ý
0gradients_1/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients_1/logistic_loss_2/Select_grad/zeros_like=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
8gradients_1/logistic_loss_2/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_2/Select_grad/Select1^gradients_1/logistic_loss_2/Select_grad/Select_1
Ź
@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_2/Select_grad/Select9^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select
˛
Bgradients_1/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_2/Select_grad/Select_19^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
,gradients_1/logistic_loss_2/mul_grad/Shape_1Shapeones_like_1*
_output_shapes
:*
T0*
out_type0
ę
:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/mul_grad/Shape,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ż
(gradients_1/logistic_loss_2/mul_grad/MulMul?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1ones_like_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
(gradients_1/logistic_loss_2/mul_grad/SumSum(gradients_1/logistic_loss_2/mul_grad/Mul:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Í
,gradients_1/logistic_loss_2/mul_grad/ReshapeReshape(gradients_1/logistic_loss_2/mul_grad/Sum*gradients_1/logistic_loss_2/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
*gradients_1/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
*gradients_1/logistic_loss_2/mul_grad/Sum_1Sum*gradients_1/logistic_loss_2/mul_grad/Mul_1<gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ó
.gradients_1/logistic_loss_2/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_2/mul_grad/Sum_1,gradients_1/logistic_loss_2/mul_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

5gradients_1/logistic_loss_2/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/mul_grad/Reshape/^gradients_1/logistic_loss_2/mul_grad/Reshape_1
˘
=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/mul_grad/Reshape6^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/mul_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
?gradients_1/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/mul_grad/Reshape_16^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˘
(gradients_1/logistic_loss_2/Exp_grad/mulMul*gradients_1/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

4gradients_1/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
0gradients_1/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual(gradients_1/logistic_loss_2/Exp_grad/mul4gradients_1/logistic_loss_2/Select_1_grad/zeros_like*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ě
2gradients_1/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual4gradients_1/logistic_loss_2/Select_1_grad/zeros_like(gradients_1/logistic_loss_2/Exp_grad/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ş
:gradients_1/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_2/Select_1_grad/Select3^gradients_1/logistic_loss_2/Select_1_grad/Select_1
´
Bgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_2/Select_1_grad/Select;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_1_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
Dgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_2/Select_1_grad/Select_1;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/logistic_loss_2/Select_1_grad/Select_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ľ
(gradients_1/logistic_loss_2/Neg_grad/NegNegBgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/AddNAddN@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_2/Neg_grad/Neg*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes
:

>gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN:^gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad

Fgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
Hgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
ţ
3gradients_1/Discriminator/prob_1/MatMul_grad/MatMulMatMulFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(

5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
ł
=gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul6^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1
Á
Egradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Discriminator/prob_1/MatMul_grad/MatMul>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
Ggradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	
Š
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
´
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ĺ
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
:˙˙˙˙˙˙˙˙˙
ë
Egradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ś
Ngradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ę
?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
Agradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SumSum?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectNgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1SumAgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1Pgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Bgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
Igradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOpA^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeC^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
ó
Qgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeJ^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ů
Sgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityBgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1J^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
¸
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
˛
Rgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulRgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ţ
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
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
:˙˙˙˙˙˙˙˙˙
ĺ
Mgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpE^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeG^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
ń
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
:˙˙˙˙˙˙˙˙˙
Ű
gradients_1/AddN_1AddNSgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ą
Qgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:
Ç
Vgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1R^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
á
^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1W^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¤
`gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityQgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ć
Kgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
°
Mgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ű
Ugradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpL^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulN^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
Ą
]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityKgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulV^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

_gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityMgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1V^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

§
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
˛
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ü
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
:˙˙˙˙˙˙˙˙˙
č
Dgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
Mgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ß
>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SumSum>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectMgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
:˙˙˙˙˙˙˙˙˙
Ö
Hgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp@^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeB^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
ď
Pgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeI^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape
ő
Rgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityAgradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1I^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
ś
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ż
Qgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
ű
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Sgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Egradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
Lgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpD^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeF^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
í
Tgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeM^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*V
_classL
JHloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape

Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityEgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1M^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
gradients_1/AddN_2AddNRgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
Pgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:
Ĺ
Ugradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2Q^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
Ţ
]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2V^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradV^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ă
Jgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ą
Lgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ř
Tgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpK^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulM^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1

\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityJgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulU^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityLgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1U^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

ë
3gradients_1/Generator/fake_image/Tanh_grad/TanhGradTanhGradGenerator/fake_image/Tanh\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ş
9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:
¸
>gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad4^gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
Ă
Fgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*F
_class<
:8loc:@gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ä
Hgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

3gradients_1/Generator/fake_image/MatMul_grad/MatMulMatMulFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency Generator/fake_image/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ů
5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1MatMulGenerator/last_layer/leaky_reluFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
ł
=gradients_1/Generator/fake_image/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Generator/fake_image/MatMul_grad/MatMul6^gradients_1/Generator/fake_image/MatMul_grad/MatMul_1
Á
Egradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/MatMul_grad/MatMul>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ż
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
°
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
˝
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2ShapeEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ű
6gradients_1/Generator/last_layer/leaky_relu_grad/zerosFill8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Fgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Generator/last_layer/leaky_relu_grad/Shape8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
˛
7gradients_1/Generator/last_layer/leaky_relu_grad/SelectSelect=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency6gradients_1/Generator/last_layer/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
´
9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Select=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqual6gradients_1/Generator/last_layer/leaky_relu_grad/zerosEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
4gradients_1/Generator/last_layer/leaky_relu_grad/SumSum7gradients_1/Generator/last_layer/leaky_relu_grad/SelectFgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ň
8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Generator/last_layer/leaky_relu_grad/Sum6gradients_1/Generator/last_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Hgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ř
:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_18gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Agradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape;^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1
Ó
Igradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeB^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*K
_classA
?=loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape
Ů
Kgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1B^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
´
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

Jgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
÷
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulMulIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

8gradients_1/Generator/last_layer/leaky_relu/mul_grad/SumSum8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulJgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ě
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ć
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Mul%Generator/last_layer/leaky_relu/alphaIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
Egradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
Ń
Mgradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*O
_classE
CAloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape
é
Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
gradients_1/AddN_3AddNKgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
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
Ů
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_3_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
˝
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_3agradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ś
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
ˇ
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
ť
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
Ů
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ł
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˝
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Generator/last_layer/fully_connected/BiasAddbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ś
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
Tshape0*
_output_shapes	
:*
T0

Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
ˇ
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:*
T0
Ţ
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg
ť
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:

bgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:
ů
Igradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Ngradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad
°
Vgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
˘
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
¨
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
°
Cgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
ă
Mgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1

Ugradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*V
_classL
JHloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˙
Wgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*X
_classN
LJloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1
ý
gradients_1/AddN_4AddNdgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
Á
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_43Generator/last_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
Č
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_48Generator/last_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:*
T0
ţ
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1

`gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:
 
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:

7gradients_1/Generator/third_layer/leaky_relu_grad/ShapeShape$Generator/third_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
˛
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Î
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
ţ
7gradients_1/Generator/third_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ggradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/third_layer/leaky_relu_grad/Shape9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ĺ
8gradients_1/Generator/third_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/third_layer/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/third_layer/leaky_relu_grad/zerosUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
5gradients_1/Generator/third_layer/leaky_relu_grad/SumSum8gradients_1/Generator/third_layer/leaky_relu_grad/SelectGgradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ő
9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/third_layer/leaky_relu_grad/Sum7gradients_1/Generator/third_layer/leaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Igradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ű
;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_19gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Bgradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
×
Jgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
Lgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
ś
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:

Kgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ú
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_1/Generator/third_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ď
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
é
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/third_layer/leaky_relu/alphaJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
Fgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
Ő
Ngradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*P
_classF
DBloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape
í
Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ć
gradients_1/AddN_5AddNLgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape9Generator/third_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:

Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ü
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_5`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ŕ
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_5bgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
š
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
ť
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
egradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
˝
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape-Generator/third_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:

Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ü
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ś
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency7Generator/third_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumSumNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ŕ
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul-Generator/third_layer/fully_connected/BiasAddcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1bgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
š
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
Tshape0*
_output_shapes	
:*
T0

[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
ť
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape
´
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
ŕ
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegNegegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpf^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1M^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg
ż
agradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
 
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:
ű
Jgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradcgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Ogradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpd^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyK^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad
´
Wgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitycgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape

Ygradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
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
Ś
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
Ź
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
ł
Dgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(

Fgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1MatMul!Generator/second_layer/leaky_reluWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
ć
Ngradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1

Vgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Xgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*Y
_classO
MKloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1

gradients_1/AddN_6AddNegradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
Ă
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_64Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
Ę
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_69Generator/third_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:

Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpM^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1

agradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul
¤
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
´
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Đ
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
:˙˙˙˙˙˙˙˙˙
ĺ
?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Hgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/Generator/second_layer/leaky_relu_grad/Shape:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
É
9gradients_1/Generator/second_layer/leaky_relu_grad/SelectSelect?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency8gradients_1/Generator/second_layer/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ë
;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Select?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqual8gradients_1/Generator/second_layer/leaky_relu_grad/zerosVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

6gradients_1/Generator/second_layer/leaky_relu_grad/SumSum9gradients_1/Generator/second_layer/leaky_relu_grad/SelectHgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ř
:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeReshape6gradients_1/Generator/second_layer/leaky_relu_grad/Sum8gradients_1/Generator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1Sum;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Jgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ţ
<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1Reshape8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
Cgradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_depsNoOp;^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape=^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1
Ű
Kgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeD^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
Mgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1D^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
¸
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
 
Lgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ý
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulMulKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

:gradients_1/Generator/second_layer/leaky_relu/mul_grad/SumSum:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulLgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ň
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeReshape:gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
ě
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Mul'Generator/second_layer/leaky_relu/alphaKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Ngradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ó
Ggradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp?^gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeA^gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
Ů
Ogradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeH^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*Q
_classG
ECloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape
ń
Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1H^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
gradients_1/AddN_7AddNMgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*
N
Ë
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape:Generator/second_layer/batch_normalization/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0

Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
ß
agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_7agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ă
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_7cgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ź
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
ż
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
ż
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape.Generator/second_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0

Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
ß
agradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Š
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mulagradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ă
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul.Generator/second_layer/fully_connected/BiasAdddgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1cgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ź
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
Tshape0*
_output_shapes	
:*
T0

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
ż
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:*
T0
â
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/NegNegfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:

Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpg^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1N^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
Ă
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
¤
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:
ý
Kgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGraddgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:

Pgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpe^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyL^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
¸
Xgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitydgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
Ą
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Muldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1;Generator/second_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:

\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulR^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
Ş
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
°
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
ś
Egradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulMatMulXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency2Generator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(

Ggradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/first_layer/leaky_reluXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
é
Ogradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpF^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulH^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1

Wgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityEgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulP^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ygradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityGgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1P^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:


gradients_1/AddN_8AddNfgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*
_output_shapes	
:*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
Ĺ
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_85Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:
Ě
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_8:Generator/second_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:

Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpN^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
˘
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:
¨
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:

7gradients_1/Generator/first_layer/leaky_relu_grad/ShapeShape$Generator/first_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ś
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Đ
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2ShapeWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
ţ
7gradients_1/Generator/first_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ö
>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ggradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/first_layer/leaky_relu_grad/Shape9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ç
8gradients_1/Generator/first_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
É
:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/first_layer/leaky_relu_grad/zerosWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˙
5gradients_1/Generator/first_layer/leaky_relu_grad/SumSum8gradients_1/Generator/first_layer/leaky_relu_grad/SelectGgradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ő
9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/first_layer/leaky_relu_grad/Sum7gradients_1/Generator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Igradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ű
;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_19gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Bgradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
×
Jgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
Lgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ş
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:

Kgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
î
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients_1/Generator/first_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ď
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
é
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/first_layer/leaky_relu/alphaJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
Fgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
Ő
Ngradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
í
Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*R
_classH
FDloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
Ć
gradients_1/AddN_9AddNLgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*
N
Ş
Jgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
data_formatNHWC*
_output_shapes	
:*
T0
š
Ogradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_9K^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
Ě
Wgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_9P^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ygradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*]
_classS
QOloc:@gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
˛
Dgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/first_layer/fully_connected/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d*
transpose_a( *
transpose_b(

Fgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise_inWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d*
transpose_a(*
transpose_b( 
ć
Ngradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1

Vgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
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
Â
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
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
 *wž?*
dtype0*
_output_shapes
: 

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
Â
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
|
beta2_power_1/readIdentitybeta2_power_1*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: *
T0
ĺ
SGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
Ď
IGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
č
CGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d
č
1Generator/first_layer/fully_connected/kernel/Adam
VariableV2*
_output_shapes
:	d*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d*
dtype0
Î
8Generator/first_layer/fully_connected/kernel/Adam/AssignAssign1Generator/first_layer/fully_connected/kernel/AdamCGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
ŕ
6Generator/first_layer/fully_connected/kernel/Adam/readIdentity1Generator/first_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
ç
UGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d      
Ń
KGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
î
EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d
ę
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
Ô
:Generator/first_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/first_layer/fully_connected/kernel/Adam_1EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0
ä
8Generator/first_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/first_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d
Ď
AGenerator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ü
/Generator/first_layer/fully_connected/bias/Adam
VariableV2*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Â
6Generator/first_layer/fully_connected/bias/Adam/AssignAssign/Generator/first_layer/fully_connected/bias/AdamAGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ö
4Generator/first_layer/fully_connected/bias/Adam/readIdentity/Generator/first_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:
Ń
CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ţ
1Generator/first_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:
Č
8Generator/first_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/first_layer/fully_connected/bias/Adam_1CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ú
6Generator/first_layer/fully_connected/bias/Adam_1/readIdentity1Generator/first_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:
ç
TGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0
Ń
JGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
í
DGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillTGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorJGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0
ě
2Generator/second_layer/fully_connected/kernel/Adam
VariableV2* 
_output_shapes
:
*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
*
dtype0
Ó
9Generator/second_layer/fully_connected/kernel/Adam/AssignAssign2Generator/second_layer/fully_connected/kernel/AdamDGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(
ä
7Generator/second_layer/fully_connected/kernel/Adam/readIdentity2Generator/second_layer/fully_connected/kernel/Adam*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

é
VGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ó
LGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ó
FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillVGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
*
T0
î
4Generator/second_layer/fully_connected/kernel/Adam_1
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
Ů
;Generator/second_layer/fully_connected/kernel/Adam_1/AssignAssign4Generator/second_layer/fully_connected/kernel/Adam_1FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

č
9Generator/second_layer/fully_connected/kernel/Adam_1/readIdentity4Generator/second_layer/fully_connected/kernel/Adam_1*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:

Ń
BGenerator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    
Ţ
0Generator/second_layer/fully_connected/bias/Adam
VariableV2*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ć
7Generator/second_layer/fully_connected/bias/Adam/AssignAssign0Generator/second_layer/fully_connected/bias/AdamBGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ů
5Generator/second_layer/fully_connected/bias/Adam/readIdentity0Generator/second_layer/fully_connected/bias/Adam*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:
Ó
DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ŕ
2Generator/second_layer/fully_connected/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container 
Ě
9Generator/second_layer/fully_connected/bias/Adam_1/AssignAssign2Generator/second_layer/fully_connected/bias/Adam_1DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ý
7Generator/second_layer/fully_connected/bias/Adam_1/readIdentity2Generator/second_layer/fully_connected/bias/Adam_1*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:*
T0
Ű
GGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
č
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
Ú
<Generator/second_layer/batch_normalization/gamma/Adam/AssignAssign5Generator/second_layer/batch_normalization/gamma/AdamGGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
č
:Generator/second_layer/batch_normalization/gamma/Adam/readIdentity5Generator/second_layer/batch_normalization/gamma/Adam*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:*
T0
Ý
IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ę
7Generator/second_layer/batch_normalization/gamma/Adam_1
VariableV2*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
ŕ
>Generator/second_layer/batch_normalization/gamma/Adam_1/AssignAssign7Generator/second_layer/batch_normalization/gamma/Adam_1IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(
ě
<Generator/second_layer/batch_normalization/gamma/Adam_1/readIdentity7Generator/second_layer/batch_normalization/gamma/Adam_1*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:
Ů
FGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ć
4Generator/second_layer/batch_normalization/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:
Ö
;Generator/second_layer/batch_normalization/beta/Adam/AssignAssign4Generator/second_layer/batch_normalization/beta/AdamFGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(
ĺ
9Generator/second_layer/batch_normalization/beta/Adam/readIdentity4Generator/second_layer/batch_normalization/beta/Adam*
_output_shapes	
:*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
Ű
HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
č
6Generator/second_layer/batch_normalization/beta/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:
Ü
=Generator/second_layer/batch_normalization/beta/Adam_1/AssignAssign6Generator/second_layer/batch_normalization/beta/Adam_1HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
é
;Generator/second_layer/batch_normalization/beta/Adam_1/readIdentity6Generator/second_layer/batch_normalization/beta/Adam_1*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:
ĺ
SGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ď
IGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
é
CGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ę
1Generator/third_layer/fully_connected/kernel/Adam
VariableV2*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ď
8Generator/third_layer/fully_connected/kernel/Adam/AssignAssign1Generator/third_layer/fully_connected/kernel/AdamCGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:

á
6Generator/third_layer/fully_connected/kernel/Adam/readIdentity1Generator/third_layer/fully_connected/kernel/Adam* 
_output_shapes
:
*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
ç
UGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Ń
KGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0
ď
EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ě
3Generator/third_layer/fully_connected/kernel/Adam_1
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
Ő
:Generator/third_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/third_layer/fully_connected/kernel/Adam_1EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ĺ
8Generator/third_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/third_layer/fully_connected/kernel/Adam_1*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
*
T0
Ď
AGenerator/third_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ü
/Generator/third_layer/fully_connected/bias/Adam
VariableV2*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Â
6Generator/third_layer/fully_connected/bias/Adam/AssignAssign/Generator/third_layer/fully_connected/bias/AdamAGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
Ö
4Generator/third_layer/fully_connected/bias/Adam/readIdentity/Generator/third_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:
Ń
CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ţ
1Generator/third_layer/fully_connected/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container 
Č
8Generator/third_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/third_layer/fully_connected/bias/Adam_1CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ú
6Generator/third_layer/fully_connected/bias/Adam_1/readIdentity1Generator/third_layer/fully_connected/bias/Adam_1*
_output_shapes	
:*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
Ů
FGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*    
ć
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
Ö
;Generator/third_layer/batch_normalization/gamma/Adam/AssignAssign4Generator/third_layer/batch_normalization/gamma/AdamFGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
ĺ
9Generator/third_layer/batch_normalization/gamma/Adam/readIdentity4Generator/third_layer/batch_normalization/gamma/Adam*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
Ű
HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB*    *
dtype0*
_output_shapes	
:
č
6Generator/third_layer/batch_normalization/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:
Ü
=Generator/third_layer/batch_normalization/gamma/Adam_1/AssignAssign6Generator/third_layer/batch_normalization/gamma/Adam_1HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
é
;Generator/third_layer/batch_normalization/gamma/Adam_1/readIdentity6Generator/third_layer/batch_normalization/gamma/Adam_1*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:
×
EGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ä
3Generator/third_layer/batch_normalization/beta/Adam
VariableV2*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ň
:Generator/third_layer/batch_normalization/beta/Adam/AssignAssign3Generator/third_layer/batch_normalization/beta/AdamEGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:
â
8Generator/third_layer/batch_normalization/beta/Adam/readIdentity3Generator/third_layer/batch_normalization/beta/Adam*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:
Ů
GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB*    *
dtype0*
_output_shapes	
:
ć
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
Ř
<Generator/third_layer/batch_normalization/beta/Adam_1/AssignAssign5Generator/third_layer/batch_normalization/beta/Adam_1GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ć
:Generator/third_layer/batch_normalization/beta/Adam_1/readIdentity5Generator/third_layer/batch_normalization/beta/Adam_1*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:
ă
RGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
Í
HGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ĺ
BGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zerosFillRGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

č
0Generator/last_layer/fully_connected/kernel/Adam
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
Ë
7Generator/last_layer/fully_connected/kernel/Adam/AssignAssign0Generator/last_layer/fully_connected/kernel/AdamBGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(
Ţ
5Generator/last_layer/fully_connected/kernel/Adam/readIdentity0Generator/last_layer/fully_connected/kernel/Adam*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

ĺ
TGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      
Ď
JGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ë
DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillTGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorJGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:

ę
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
Ń
9Generator/last_layer/fully_connected/kernel/Adam_1/AssignAssign2Generator/last_layer/fully_connected/kernel/Adam_1DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
â
7Generator/last_layer/fully_connected/kernel/Adam_1/readIdentity2Generator/last_layer/fully_connected/kernel/Adam_1*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:

Ů
PGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
É
FGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    
Ř
@Generator/last_layer/fully_connected/bias/Adam/Initializer/zerosFillPGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorFGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:
Ú
.Generator/last_layer/fully_connected/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container 
ž
5Generator/last_layer/fully_connected/bias/Adam/AssignAssign.Generator/last_layer/fully_connected/bias/Adam@Generator/last_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
Ó
3Generator/last_layer/fully_connected/bias/Adam/readIdentity.Generator/last_layer/fully_connected/bias/Adam*
_output_shapes	
:*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
Ű
RGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:*
dtype0
Ë
HGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Ţ
BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zerosFillRGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:
Ü
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
Ä
7Generator/last_layer/fully_connected/bias/Adam_1/AssignAssign0Generator/last_layer/fully_connected/bias/Adam_1BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:
×
5Generator/last_layer/fully_connected/bias/Adam_1/readIdentity0Generator/last_layer/fully_connected/bias/Adam_1*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:
ă
UGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0*
_output_shapes
:
Ó
KGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ě
EGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zerosFillUGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorKGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/Const*
_output_shapes	
:*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
ä
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
Ň
:Generator/last_layer/batch_normalization/gamma/Adam/AssignAssign3Generator/last_layer/batch_normalization/gamma/AdamEGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:
â
8Generator/last_layer/batch_normalization/gamma/Adam/readIdentity3Generator/last_layer/batch_normalization/gamma/Adam*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:
ĺ
WGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:*
dtype0
Ő
MGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
ň
GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zerosFillWGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorMGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:
ć
5Generator/last_layer/batch_normalization/gamma/Adam_1
VariableV2*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
Ř
<Generator/last_layer/batch_normalization/gamma/Adam_1/AssignAssign5Generator/last_layer/batch_normalization/gamma/Adam_1GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
ć
:Generator/last_layer/batch_normalization/gamma/Adam_1/readIdentity5Generator/last_layer/batch_normalization/gamma/Adam_1*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:
á
TGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:*
dtype0*
_output_shapes
:
Ń
JGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
č
DGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zerosFillTGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorJGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:
â
2Generator/last_layer/batch_normalization/beta/Adam
VariableV2*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
Î
9Generator/last_layer/batch_normalization/beta/Adam/AssignAssign2Generator/last_layer/batch_normalization/beta/AdamDGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(
ß
7Generator/last_layer/batch_normalization/beta/Adam/readIdentity2Generator/last_layer/batch_normalization/beta/Adam*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:
ă
VGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:*
dtype0*
_output_shapes
:
Ó
LGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    
î
FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zerosFillVGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0
ä
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
Ô
;Generator/last_layer/batch_normalization/beta/Adam_1/AssignAssign4Generator/last_layer/batch_normalization/beta/Adam_1FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
ă
9Generator/last_layer/batch_normalization/beta/Adam_1/readIdentity4Generator/last_layer/batch_normalization/beta/Adam_1*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:
Ă
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
Ľ
2Generator/fake_image/kernel/Adam/Initializer/zerosFillBGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensor8Generator/fake_image/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:

Č
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
'Generator/fake_image/kernel/Adam/AssignAssign Generator/fake_image/kernel/Adam2Generator/fake_image/kernel/Adam/Initializer/zeros*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ž
%Generator/fake_image/kernel/Adam/readIdentity Generator/fake_image/kernel/Adam*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:

Ĺ
DGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ż
:Generator/fake_image/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ť
4Generator/fake_image/kernel/Adam_1/Initializer/zerosFillDGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensor:Generator/fake_image/kernel/Adam_1/Initializer/zeros/Const*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:

Ę
"Generator/fake_image/kernel/Adam_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel
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
˛
'Generator/fake_image/kernel/Adam_1/readIdentity"Generator/fake_image/kernel/Adam_1* 
_output_shapes
:
*
T0*.
_class$
" loc:@Generator/fake_image/kernel
­
0Generator/fake_image/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0
ş
Generator/fake_image/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *,
_class"
 loc:@Generator/fake_image/bias
ţ
%Generator/fake_image/bias/Adam/AssignAssignGenerator/fake_image/bias/Adam0Generator/fake_image/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
Ł
#Generator/fake_image/bias/Adam/readIdentityGenerator/fake_image/bias/Adam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
Ż
2Generator/fake_image/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB*    *
dtype0*
_output_shapes	
:
ź
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
'Generator/fake_image/bias/Adam_1/AssignAssign Generator/fake_image/bias/Adam_12Generator/fake_image/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias
§
%Generator/fake_image/bias/Adam_1/readIdentity Generator/fake_image/bias/Adam_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:
Y
Adam_1/learning_rateConst*
valueB
 *ˇQ9*
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
 *wž?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
˝
DAdam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/first_layer/fully_connected/kernel1Generator/first_layer/fully_connected/kernel/Adam3Generator/first_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	d*
use_locking( *
T0
°
BAdam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/first_layer/fully_connected/bias/Generator/first_layer/fully_connected/bias/Adam1Generator/first_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:
Ä
EAdam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam-Generator/second_layer/fully_connected/kernel2Generator/second_layer/fully_connected/kernel/Adam4Generator/second_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:

ś
CAdam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam+Generator/second_layer/fully_connected/bias0Generator/second_layer/fully_connected/bias/Adam2Generator/second_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias
Ů
HAdam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam0Generator/second_layer/batch_normalization/gamma5Generator/second_layer/batch_normalization/gamma/Adam7Generator/second_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:
Ň
GAdam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam/Generator/second_layer/batch_normalization/beta4Generator/second_layer/batch_normalization/beta/Adam6Generator/second_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:*
use_locking( 
ž
DAdam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/third_layer/fully_connected/kernel1Generator/third_layer/fully_connected/kernel/Adam3Generator/third_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( 
°
BAdam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/third_layer/fully_connected/bias/Generator/third_layer/fully_connected/bias/Adam1Generator/third_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
Ó
GAdam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam/Generator/third_layer/batch_normalization/gamma4Generator/third_layer/batch_normalization/gamma/Adam6Generator/third_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:
Ě
FAdam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdam	ApplyAdam.Generator/third_layer/batch_normalization/beta3Generator/third_layer/batch_normalization/beta/Adam5Generator/third_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:
¸
CAdam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Generator/last_layer/fully_connected/kernel0Generator/last_layer/fully_connected/kernel/Adam2Generator/last_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
Ş
AAdam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Generator/last_layer/fully_connected/bias.Generator/last_layer/fully_connected/bias/Adam0Generator/last_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
Í
FAdam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Generator/last_layer/batch_normalization/gamma3Generator/last_layer/batch_normalization/gamma/Adam5Generator/last_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:
Ć
EAdam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Generator/last_layer/batch_normalization/beta2Generator/last_layer/batch_normalization/beta/Adam4Generator/last_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes	
:*
use_locking( *
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
use_nesterov( 
Ř
3Adam_1/update_Generator/fake_image/kernel/ApplyAdam	ApplyAdamGenerator/fake_image/kernel Generator/fake_image/kernel/Adam"Generator/fake_image/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( 
Ę
1Adam_1/update_Generator/fake_image/bias/ApplyAdam	ApplyAdamGenerator/fake_image/biasGenerator/fake_image/bias/Adam Generator/fake_image/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
use_nesterov( *
_output_shapes	
:
Ő	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
Ş
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias
×	
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta22^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: *
T0
Ž
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
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
generator_loss:0"Ú%
trainable_variablesÂ%ż%
ç
.Generator/first_layer/fully_connected/kernel:03Generator/first_layer/fully_connected/kernel/Assign3Generator/first_layer/fully_connected/kernel/read:02IGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
Ö
,Generator/first_layer/fully_connected/bias:01Generator/first_layer/fully_connected/bias/Assign1Generator/first_layer/fully_connected/bias/read:02>Generator/first_layer/fully_connected/bias/Initializer/zeros:08
ë
/Generator/second_layer/fully_connected/kernel:04Generator/second_layer/fully_connected/kernel/Assign4Generator/second_layer/fully_connected/kernel/read:02JGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
Ú
-Generator/second_layer/fully_connected/bias:02Generator/second_layer/fully_connected/bias/Assign2Generator/second_layer/fully_connected/bias/read:02?Generator/second_layer/fully_connected/bias/Initializer/zeros:08
í
2Generator/second_layer/batch_normalization/gamma:07Generator/second_layer/batch_normalization/gamma/Assign7Generator/second_layer/batch_normalization/gamma/read:02CGenerator/second_layer/batch_normalization/gamma/Initializer/ones:08
ę
1Generator/second_layer/batch_normalization/beta:06Generator/second_layer/batch_normalization/beta/Assign6Generator/second_layer/batch_normalization/beta/read:02CGenerator/second_layer/batch_normalization/beta/Initializer/zeros:08
ç
.Generator/third_layer/fully_connected/kernel:03Generator/third_layer/fully_connected/kernel/Assign3Generator/third_layer/fully_connected/kernel/read:02IGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform:08
Ö
,Generator/third_layer/fully_connected/bias:01Generator/third_layer/fully_connected/bias/Assign1Generator/third_layer/fully_connected/bias/read:02>Generator/third_layer/fully_connected/bias/Initializer/zeros:08
é
1Generator/third_layer/batch_normalization/gamma:06Generator/third_layer/batch_normalization/gamma/Assign6Generator/third_layer/batch_normalization/gamma/read:02BGenerator/third_layer/batch_normalization/gamma/Initializer/ones:08
ć
0Generator/third_layer/batch_normalization/beta:05Generator/third_layer/batch_normalization/beta/Assign5Generator/third_layer/batch_normalization/beta/read:02BGenerator/third_layer/batch_normalization/beta/Initializer/zeros:08
ă
-Generator/last_layer/fully_connected/kernel:02Generator/last_layer/fully_connected/kernel/Assign2Generator/last_layer/fully_connected/kernel/read:02HGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform:08
Ň
+Generator/last_layer/fully_connected/bias:00Generator/last_layer/fully_connected/bias/Assign0Generator/last_layer/fully_connected/bias/read:02=Generator/last_layer/fully_connected/bias/Initializer/zeros:08
ĺ
0Generator/last_layer/batch_normalization/gamma:05Generator/last_layer/batch_normalization/gamma/Assign5Generator/last_layer/batch_normalization/gamma/read:02AGenerator/last_layer/batch_normalization/gamma/Initializer/ones:08
â
/Generator/last_layer/batch_normalization/beta:04Generator/last_layer/batch_normalization/beta/Assign4Generator/last_layer/batch_normalization/beta/read:02AGenerator/last_layer/batch_normalization/beta/Initializer/zeros:08
Ł
Generator/fake_image/kernel:0"Generator/fake_image/kernel/Assign"Generator/fake_image/kernel/read:028Generator/fake_image/kernel/Initializer/random_uniform:08

Generator/fake_image/bias:0 Generator/fake_image/bias/Assign Generator/fake_image/bias/read:02-Generator/fake_image/bias/Initializer/zeros:08
÷
2Discriminator/first_layer/fully_connected/kernel:07Discriminator/first_layer/fully_connected/kernel/Assign7Discriminator/first_layer/fully_connected/kernel/read:02MDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
ć
0Discriminator/first_layer/fully_connected/bias:05Discriminator/first_layer/fully_connected/bias/Assign5Discriminator/first_layer/fully_connected/bias/read:02BDiscriminator/first_layer/fully_connected/bias/Initializer/zeros:08
ű
3Discriminator/second_layer/fully_connected/kernel:08Discriminator/second_layer/fully_connected/kernel/Assign8Discriminator/second_layer/fully_connected/kernel/read:02NDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
ę
1Discriminator/second_layer/fully_connected/bias:06Discriminator/second_layer/fully_connected/bias/Assign6Discriminator/second_layer/fully_connected/bias/read:02CDiscriminator/second_layer/fully_connected/bias/Initializer/zeros:08

Discriminator/prob/kernel:0 Discriminator/prob/kernel/Assign Discriminator/prob/kernel/read:026Discriminator/prob/kernel/Initializer/random_uniform:08

Discriminator/prob/bias:0Discriminator/prob/bias/AssignDiscriminator/prob/bias/read:02+Discriminator/prob/bias/Initializer/zeros:08"
train_op

Adam
Adam_1"ľ
	variablesŚ˘
ç
.Generator/first_layer/fully_connected/kernel:03Generator/first_layer/fully_connected/kernel/Assign3Generator/first_layer/fully_connected/kernel/read:02IGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
Ö
,Generator/first_layer/fully_connected/bias:01Generator/first_layer/fully_connected/bias/Assign1Generator/first_layer/fully_connected/bias/read:02>Generator/first_layer/fully_connected/bias/Initializer/zeros:08
ë
/Generator/second_layer/fully_connected/kernel:04Generator/second_layer/fully_connected/kernel/Assign4Generator/second_layer/fully_connected/kernel/read:02JGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
Ú
-Generator/second_layer/fully_connected/bias:02Generator/second_layer/fully_connected/bias/Assign2Generator/second_layer/fully_connected/bias/read:02?Generator/second_layer/fully_connected/bias/Initializer/zeros:08
í
2Generator/second_layer/batch_normalization/gamma:07Generator/second_layer/batch_normalization/gamma/Assign7Generator/second_layer/batch_normalization/gamma/read:02CGenerator/second_layer/batch_normalization/gamma/Initializer/ones:08
ę
1Generator/second_layer/batch_normalization/beta:06Generator/second_layer/batch_normalization/beta/Assign6Generator/second_layer/batch_normalization/beta/read:02CGenerator/second_layer/batch_normalization/beta/Initializer/zeros:08

8Generator/second_layer/batch_normalization/moving_mean:0=Generator/second_layer/batch_normalization/moving_mean/Assign=Generator/second_layer/batch_normalization/moving_mean/read:02JGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros:0

<Generator/second_layer/batch_normalization/moving_variance:0AGenerator/second_layer/batch_normalization/moving_variance/AssignAGenerator/second_layer/batch_normalization/moving_variance/read:02MGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones:0
ç
.Generator/third_layer/fully_connected/kernel:03Generator/third_layer/fully_connected/kernel/Assign3Generator/third_layer/fully_connected/kernel/read:02IGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform:08
Ö
,Generator/third_layer/fully_connected/bias:01Generator/third_layer/fully_connected/bias/Assign1Generator/third_layer/fully_connected/bias/read:02>Generator/third_layer/fully_connected/bias/Initializer/zeros:08
é
1Generator/third_layer/batch_normalization/gamma:06Generator/third_layer/batch_normalization/gamma/Assign6Generator/third_layer/batch_normalization/gamma/read:02BGenerator/third_layer/batch_normalization/gamma/Initializer/ones:08
ć
0Generator/third_layer/batch_normalization/beta:05Generator/third_layer/batch_normalization/beta/Assign5Generator/third_layer/batch_normalization/beta/read:02BGenerator/third_layer/batch_normalization/beta/Initializer/zeros:08

7Generator/third_layer/batch_normalization/moving_mean:0<Generator/third_layer/batch_normalization/moving_mean/Assign<Generator/third_layer/batch_normalization/moving_mean/read:02IGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros:0

;Generator/third_layer/batch_normalization/moving_variance:0@Generator/third_layer/batch_normalization/moving_variance/Assign@Generator/third_layer/batch_normalization/moving_variance/read:02LGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones:0
ă
-Generator/last_layer/fully_connected/kernel:02Generator/last_layer/fully_connected/kernel/Assign2Generator/last_layer/fully_connected/kernel/read:02HGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform:08
Ň
+Generator/last_layer/fully_connected/bias:00Generator/last_layer/fully_connected/bias/Assign0Generator/last_layer/fully_connected/bias/read:02=Generator/last_layer/fully_connected/bias/Initializer/zeros:08
ĺ
0Generator/last_layer/batch_normalization/gamma:05Generator/last_layer/batch_normalization/gamma/Assign5Generator/last_layer/batch_normalization/gamma/read:02AGenerator/last_layer/batch_normalization/gamma/Initializer/ones:08
â
/Generator/last_layer/batch_normalization/beta:04Generator/last_layer/batch_normalization/beta/Assign4Generator/last_layer/batch_normalization/beta/read:02AGenerator/last_layer/batch_normalization/beta/Initializer/zeros:08
ü
6Generator/last_layer/batch_normalization/moving_mean:0;Generator/last_layer/batch_normalization/moving_mean/Assign;Generator/last_layer/batch_normalization/moving_mean/read:02HGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros:0

:Generator/last_layer/batch_normalization/moving_variance:0?Generator/last_layer/batch_normalization/moving_variance/Assign?Generator/last_layer/batch_normalization/moving_variance/read:02KGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones:0
Ł
Generator/fake_image/kernel:0"Generator/fake_image/kernel/Assign"Generator/fake_image/kernel/read:028Generator/fake_image/kernel/Initializer/random_uniform:08

Generator/fake_image/bias:0 Generator/fake_image/bias/Assign Generator/fake_image/bias/read:02-Generator/fake_image/bias/Initializer/zeros:08
÷
2Discriminator/first_layer/fully_connected/kernel:07Discriminator/first_layer/fully_connected/kernel/Assign7Discriminator/first_layer/fully_connected/kernel/read:02MDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
ć
0Discriminator/first_layer/fully_connected/bias:05Discriminator/first_layer/fully_connected/bias/Assign5Discriminator/first_layer/fully_connected/bias/read:02BDiscriminator/first_layer/fully_connected/bias/Initializer/zeros:08
ű
3Discriminator/second_layer/fully_connected/kernel:08Discriminator/second_layer/fully_connected/kernel/Assign8Discriminator/second_layer/fully_connected/kernel/read:02NDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
ę
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
ř
5Discriminator/first_layer/fully_connected/bias/Adam:0:Discriminator/first_layer/fully_connected/bias/Adam/Assign:Discriminator/first_layer/fully_connected/bias/Adam/read:02GDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros:0

7Discriminator/first_layer/fully_connected/bias/Adam_1:0<Discriminator/first_layer/fully_connected/bias/Adam_1/Assign<Discriminator/first_layer/fully_connected/bias/Adam_1/read:02IDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros:0

8Discriminator/second_layer/fully_connected/kernel/Adam:0=Discriminator/second_layer/fully_connected/kernel/Adam/Assign=Discriminator/second_layer/fully_connected/kernel/Adam/read:02JDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros:0

:Discriminator/second_layer/fully_connected/kernel/Adam_1:0?Discriminator/second_layer/fully_connected/kernel/Adam_1/Assign?Discriminator/second_layer/fully_connected/kernel/Adam_1/read:02LDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ü
6Discriminator/second_layer/fully_connected/bias/Adam:0;Discriminator/second_layer/fully_connected/bias/Adam/Assign;Discriminator/second_layer/fully_connected/bias/Adam/read:02HDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros:0

8Discriminator/second_layer/fully_connected/bias/Adam_1:0=Discriminator/second_layer/fully_connected/bias/Adam_1/Assign=Discriminator/second_layer/fully_connected/bias/Adam_1/read:02JDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
¤
 Discriminator/prob/kernel/Adam:0%Discriminator/prob/kernel/Adam/Assign%Discriminator/prob/kernel/Adam/read:022Discriminator/prob/kernel/Adam/Initializer/zeros:0
Ź
"Discriminator/prob/kernel/Adam_1:0'Discriminator/prob/kernel/Adam_1/Assign'Discriminator/prob/kernel/Adam_1/read:024Discriminator/prob/kernel/Adam_1/Initializer/zeros:0

Discriminator/prob/bias/Adam:0#Discriminator/prob/bias/Adam/Assign#Discriminator/prob/bias/Adam/read:020Discriminator/prob/bias/Adam/Initializer/zeros:0
¤
 Discriminator/prob/bias/Adam_1:0%Discriminator/prob/bias/Adam_1/Assign%Discriminator/prob/bias/Adam_1/read:022Discriminator/prob/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
đ
3Generator/first_layer/fully_connected/kernel/Adam:08Generator/first_layer/fully_connected/kernel/Adam/Assign8Generator/first_layer/fully_connected/kernel/Adam/read:02EGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ř
5Generator/first_layer/fully_connected/kernel/Adam_1:0:Generator/first_layer/fully_connected/kernel/Adam_1/Assign:Generator/first_layer/fully_connected/kernel/Adam_1/read:02GGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
č
1Generator/first_layer/fully_connected/bias/Adam:06Generator/first_layer/fully_connected/bias/Adam/Assign6Generator/first_layer/fully_connected/bias/Adam/read:02CGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros:0
đ
3Generator/first_layer/fully_connected/bias/Adam_1:08Generator/first_layer/fully_connected/bias/Adam_1/Assign8Generator/first_layer/fully_connected/bias/Adam_1/read:02EGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
ô
4Generator/second_layer/fully_connected/kernel/Adam:09Generator/second_layer/fully_connected/kernel/Adam/Assign9Generator/second_layer/fully_connected/kernel/Adam/read:02FGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ü
6Generator/second_layer/fully_connected/kernel/Adam_1:0;Generator/second_layer/fully_connected/kernel/Adam_1/Assign;Generator/second_layer/fully_connected/kernel/Adam_1/read:02HGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ě
2Generator/second_layer/fully_connected/bias/Adam:07Generator/second_layer/fully_connected/bias/Adam/Assign7Generator/second_layer/fully_connected/bias/Adam/read:02DGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros:0
ô
4Generator/second_layer/fully_connected/bias/Adam_1:09Generator/second_layer/fully_connected/bias/Adam_1/Assign9Generator/second_layer/fully_connected/bias/Adam_1/read:02FGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros:0

7Generator/second_layer/batch_normalization/gamma/Adam:0<Generator/second_layer/batch_normalization/gamma/Adam/Assign<Generator/second_layer/batch_normalization/gamma/Adam/read:02IGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros:0

9Generator/second_layer/batch_normalization/gamma/Adam_1:0>Generator/second_layer/batch_normalization/gamma/Adam_1/Assign>Generator/second_layer/batch_normalization/gamma/Adam_1/read:02KGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
ü
6Generator/second_layer/batch_normalization/beta/Adam:0;Generator/second_layer/batch_normalization/beta/Adam/Assign;Generator/second_layer/batch_normalization/beta/Adam/read:02HGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros:0

8Generator/second_layer/batch_normalization/beta/Adam_1:0=Generator/second_layer/batch_normalization/beta/Adam_1/Assign=Generator/second_layer/batch_normalization/beta/Adam_1/read:02JGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
đ
3Generator/third_layer/fully_connected/kernel/Adam:08Generator/third_layer/fully_connected/kernel/Adam/Assign8Generator/third_layer/fully_connected/kernel/Adam/read:02EGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ř
5Generator/third_layer/fully_connected/kernel/Adam_1:0:Generator/third_layer/fully_connected/kernel/Adam_1/Assign:Generator/third_layer/fully_connected/kernel/Adam_1/read:02GGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
č
1Generator/third_layer/fully_connected/bias/Adam:06Generator/third_layer/fully_connected/bias/Adam/Assign6Generator/third_layer/fully_connected/bias/Adam/read:02CGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros:0
đ
3Generator/third_layer/fully_connected/bias/Adam_1:08Generator/third_layer/fully_connected/bias/Adam_1/Assign8Generator/third_layer/fully_connected/bias/Adam_1/read:02EGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
ü
6Generator/third_layer/batch_normalization/gamma/Adam:0;Generator/third_layer/batch_normalization/gamma/Adam/Assign;Generator/third_layer/batch_normalization/gamma/Adam/read:02HGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros:0

8Generator/third_layer/batch_normalization/gamma/Adam_1:0=Generator/third_layer/batch_normalization/gamma/Adam_1/Assign=Generator/third_layer/batch_normalization/gamma/Adam_1/read:02JGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
ř
5Generator/third_layer/batch_normalization/beta/Adam:0:Generator/third_layer/batch_normalization/beta/Adam/Assign:Generator/third_layer/batch_normalization/beta/Adam/read:02GGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros:0

7Generator/third_layer/batch_normalization/beta/Adam_1:0<Generator/third_layer/batch_normalization/beta/Adam_1/Assign<Generator/third_layer/batch_normalization/beta/Adam_1/read:02IGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
ě
2Generator/last_layer/fully_connected/kernel/Adam:07Generator/last_layer/fully_connected/kernel/Adam/Assign7Generator/last_layer/fully_connected/kernel/Adam/read:02DGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros:0
ô
4Generator/last_layer/fully_connected/kernel/Adam_1:09Generator/last_layer/fully_connected/kernel/Adam_1/Assign9Generator/last_layer/fully_connected/kernel/Adam_1/read:02FGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
ä
0Generator/last_layer/fully_connected/bias/Adam:05Generator/last_layer/fully_connected/bias/Adam/Assign5Generator/last_layer/fully_connected/bias/Adam/read:02BGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros:0
ě
2Generator/last_layer/fully_connected/bias/Adam_1:07Generator/last_layer/fully_connected/bias/Adam_1/Assign7Generator/last_layer/fully_connected/bias/Adam_1/read:02DGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
ř
5Generator/last_layer/batch_normalization/gamma/Adam:0:Generator/last_layer/batch_normalization/gamma/Adam/Assign:Generator/last_layer/batch_normalization/gamma/Adam/read:02GGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros:0

7Generator/last_layer/batch_normalization/gamma/Adam_1:0<Generator/last_layer/batch_normalization/gamma/Adam_1/Assign<Generator/last_layer/batch_normalization/gamma/Adam_1/read:02IGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
ô
4Generator/last_layer/batch_normalization/beta/Adam:09Generator/last_layer/batch_normalization/beta/Adam/Assign9Generator/last_layer/batch_normalization/beta/Adam/read:02FGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros:0
ü
6Generator/last_layer/batch_normalization/beta/Adam_1:0;Generator/last_layer/batch_normalization/beta/Adam_1/Assign;Generator/last_layer/batch_normalization/beta/Adam_1/read:02HGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
Ź
"Generator/fake_image/kernel/Adam:0'Generator/fake_image/kernel/Adam/Assign'Generator/fake_image/kernel/Adam/read:024Generator/fake_image/kernel/Adam/Initializer/zeros:0
´
$Generator/fake_image/kernel/Adam_1:0)Generator/fake_image/kernel/Adam_1/Assign)Generator/fake_image/kernel/Adam_1/read:026Generator/fake_image/kernel/Adam_1/Initializer/zeros:0
¤
 Generator/fake_image/bias/Adam:0%Generator/fake_image/bias/Adam/Assign%Generator/fake_image/bias/Adam/read:022Generator/fake_image/bias/Adam/Initializer/zeros:0
Ź
"Generator/fake_image/bias/Adam_1:0'Generator/fake_image/bias/Adam_1/Assign'Generator/fake_image/bias/Adam_1/read:024Generator/fake_image/bias/Adam_1/Initializer/zeros:0%óHËú       áN	żT>ÉęýÖA*î
w
discriminator_loss*a	    Ďöň?    Ďöň?      đ?!    Ďöň?) śç0zö?2cIĆÂń?şP1ó?˙˙˙˙˙˙ď:              đ?        
s
generator_loss*a	    /ä?    /ä?      đ?!    /ä?) đOwŮ?2+Se*8ä?uoűpć?˙˙˙˙˙˙ď:              đ?        f