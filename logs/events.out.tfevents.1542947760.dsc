       �K"	   ����Abrain.Event:2�Y0�     ���	��:����A"��
u
Generator/noise_inPlaceholder*
dtype0*'
_output_shapes
:���������d*
shape:���������d
�
MGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
�
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&�*
dtype0
�
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&>*
dtype0*
_output_shapes
: 
�
UGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d�*

seed *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
seed2 
�
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
�
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
GGenerator/first_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
,Generator/first_layer/fully_connected/kernel
VariableV2*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�
�
3Generator/first_layer/fully_connected/kernel/AssignAssign,Generator/first_layer/fully_connected/kernelGGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0
�
1Generator/first_layer/fully_connected/kernel/readIdentity,Generator/first_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
<Generator/first_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
*Generator/first_layer/fully_connected/bias
VariableV2*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
1Generator/first_layer/fully_connected/bias/AssignAssign*Generator/first_layer/fully_connected/bias<Generator/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
/Generator/first_layer/fully_connected/bias/readIdentity*Generator/first_layer/fully_connected/bias*
_output_shapes	
:�*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
�
,Generator/first_layer/fully_connected/MatMulMatMulGenerator/noise_in1Generator/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
-Generator/first_layer/fully_connected/BiasAddBiasAdd,Generator/first_layer/fully_connected/MatMul/Generator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
k
&Generator/first_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
$Generator/first_layer/leaky_relu/mulMul&Generator/first_layer/leaky_relu/alpha-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
 Generator/first_layer/leaky_reluMaximum$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
NGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   �
�
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
�
VGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformNGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
seed2 
�
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
_output_shapes
: 
�
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulVGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
HGenerator/second_layer/fully_connected/kernel/Initializer/random_uniformAddLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
-Generator/second_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
��
�
4Generator/second_layer/fully_connected/kernel/AssignAssign-Generator/second_layer/fully_connected/kernelHGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
2Generator/second_layer/fully_connected/kernel/readIdentity-Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
�
=Generator/second_layer/fully_connected/bias/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
+Generator/second_layer/fully_connected/bias
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias
�
2Generator/second_layer/fully_connected/bias/AssignAssign+Generator/second_layer/fully_connected/bias=Generator/second_layer/fully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias
�
0Generator/second_layer/fully_connected/bias/readIdentity+Generator/second_layer/fully_connected/bias*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
-Generator/second_layer/fully_connected/MatMulMatMul Generator/first_layer/leaky_relu2Generator/second_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
.Generator/second_layer/fully_connected/BiasAddBiasAdd-Generator/second_layer/fully_connected/MatMul0Generator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
AGenerator/second_layer/batch_normalization/gamma/Initializer/onesConst*
_output_shapes	
:�*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB�*  �?*
dtype0
�
0Generator/second_layer/batch_normalization/gamma
VariableV2*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
7Generator/second_layer/batch_normalization/gamma/AssignAssign0Generator/second_layer/batch_normalization/gammaAGenerator/second_layer/batch_normalization/gamma/Initializer/ones*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
5Generator/second_layer/batch_normalization/gamma/readIdentity0Generator/second_layer/batch_normalization/gamma*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
AGenerator/second_layer/batch_normalization/beta/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
/Generator/second_layer/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:�
�
6Generator/second_layer/batch_normalization/beta/AssignAssign/Generator/second_layer/batch_normalization/betaAGenerator/second_layer/batch_normalization/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
�
4Generator/second_layer/batch_normalization/beta/readIdentity/Generator/second_layer/batch_normalization/beta*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
HGenerator/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
6Generator/second_layer/batch_normalization/moving_mean
VariableV2*
shared_name *I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
=Generator/second_layer/batch_normalization/moving_mean/AssignAssign6Generator/second_layer/batch_normalization/moving_meanHGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:�
�
;Generator/second_layer/batch_normalization/moving_mean/readIdentity6Generator/second_layer/batch_normalization/moving_mean*
_output_shapes	
:�*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean
�
KGenerator/second_layer/batch_normalization/moving_variance/Initializer/onesConst*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
:Generator/second_layer/batch_normalization/moving_variance
VariableV2*
shared_name *M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
AGenerator/second_layer/batch_normalization/moving_variance/AssignAssign:Generator/second_layer/batch_normalization/moving_varianceKGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:�
�
?Generator/second_layer/batch_normalization/moving_variance/readIdentity:Generator/second_layer/batch_normalization/moving_variance*
_output_shapes	
:�*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance

:Generator/second_layer/batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
8Generator/second_layer/batch_normalization/batchnorm/addAdd?Generator/second_layer/batch_normalization/moving_variance/read:Generator/second_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:�*
T0
�
:Generator/second_layer/batch_normalization/batchnorm/RsqrtRsqrt8Generator/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:�
�
8Generator/second_layer/batch_normalization/batchnorm/mulMul:Generator/second_layer/batch_normalization/batchnorm/Rsqrt5Generator/second_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
�
:Generator/second_layer/batch_normalization/batchnorm/mul_1Mul.Generator/second_layer/fully_connected/BiasAdd8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
:Generator/second_layer/batch_normalization/batchnorm/mul_2Mul;Generator/second_layer/batch_normalization/moving_mean/read8Generator/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:�*
T0
�
8Generator/second_layer/batch_normalization/batchnorm/subSub4Generator/second_layer/batch_normalization/beta/read:Generator/second_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
:Generator/second_layer/batch_normalization/batchnorm/add_1Add:Generator/second_layer/batch_normalization/batchnorm/mul_18Generator/second_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:����������*
T0
l
'Generator/second_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
%Generator/second_layer/leaky_relu/mulMul'Generator/second_layer/leaky_relu/alpha:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
!Generator/second_layer/leaky_reluMaximum%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
MGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *��=
�
UGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shape*

seed *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
��
�
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
_output_shapes
: 
�
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��
�
GGenerator/third_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��
�
,Generator/third_layer/fully_connected/kernel
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
�
3Generator/third_layer/fully_connected/kernel/AssignAssign,Generator/third_layer/fully_connected/kernelGGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
1Generator/third_layer/fully_connected/kernel/readIdentity,Generator/third_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��
�
<Generator/third_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB�*    
�
*Generator/third_layer/fully_connected/bias
VariableV2*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
1Generator/third_layer/fully_connected/bias/AssignAssign*Generator/third_layer/fully_connected/bias<Generator/third_layer/fully_connected/bias/Initializer/zeros*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
/Generator/third_layer/fully_connected/bias/readIdentity*Generator/third_layer/fully_connected/bias*
_output_shapes	
:�*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
�
,Generator/third_layer/fully_connected/MatMulMatMul!Generator/second_layer/leaky_relu1Generator/third_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
-Generator/third_layer/fully_connected/BiasAddBiasAdd,Generator/third_layer/fully_connected/MatMul/Generator/third_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
@Generator/third_layer/batch_normalization/gamma/Initializer/onesConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
/Generator/third_layer/batch_normalization/gamma
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container 
�
6Generator/third_layer/batch_normalization/gamma/AssignAssign/Generator/third_layer/batch_normalization/gamma@Generator/third_layer/batch_normalization/gamma/Initializer/ones*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
4Generator/third_layer/batch_normalization/gamma/readIdentity/Generator/third_layer/batch_normalization/gamma*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:�
�
@Generator/third_layer/batch_normalization/beta/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
.Generator/third_layer/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:�
�
5Generator/third_layer/batch_normalization/beta/AssignAssign.Generator/third_layer/batch_normalization/beta@Generator/third_layer/batch_normalization/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
�
3Generator/third_layer/batch_normalization/beta/readIdentity.Generator/third_layer/batch_normalization/beta*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:�
�
GGenerator/third_layer/batch_normalization/moving_mean/Initializer/zerosConst*
_output_shapes	
:�*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
valueB�*    *
dtype0
�
5Generator/third_layer/batch_normalization/moving_mean
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
	container *
shape:�
�
<Generator/third_layer/batch_normalization/moving_mean/AssignAssign5Generator/third_layer/batch_normalization/moving_meanGGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:�
�
:Generator/third_layer/batch_normalization/moving_mean/readIdentity5Generator/third_layer/batch_normalization/moving_mean*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
_output_shapes	
:�
�
JGenerator/third_layer/batch_normalization/moving_variance/Initializer/onesConst*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
9Generator/third_layer/batch_normalization/moving_variance
VariableV2*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
@Generator/third_layer/batch_normalization/moving_variance/AssignAssign9Generator/third_layer/batch_normalization/moving_varianceJGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance
�
>Generator/third_layer/batch_normalization/moving_variance/readIdentity9Generator/third_layer/batch_normalization/moving_variance*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
_output_shapes	
:�
~
9Generator/third_layer/batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
7Generator/third_layer/batch_normalization/batchnorm/addAdd>Generator/third_layer/batch_normalization/moving_variance/read9Generator/third_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:�
�
9Generator/third_layer/batch_normalization/batchnorm/RsqrtRsqrt7Generator/third_layer/batch_normalization/batchnorm/add*
_output_shapes	
:�*
T0
�
7Generator/third_layer/batch_normalization/batchnorm/mulMul9Generator/third_layer/batch_normalization/batchnorm/Rsqrt4Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
9Generator/third_layer/batch_normalization/batchnorm/mul_1Mul-Generator/third_layer/fully_connected/BiasAdd7Generator/third_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
�
9Generator/third_layer/batch_normalization/batchnorm/mul_2Mul:Generator/third_layer/batch_normalization/moving_mean/read7Generator/third_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:�*
T0
�
7Generator/third_layer/batch_normalization/batchnorm/subSub3Generator/third_layer/batch_normalization/beta/read9Generator/third_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
9Generator/third_layer/batch_normalization/batchnorm/add_1Add9Generator/third_layer/batch_normalization/batchnorm/mul_17Generator/third_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:����������*
T0
k
&Generator/third_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
$Generator/third_layer/leaky_relu/mulMul&Generator/third_layer/leaky_relu/alpha9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
 Generator/third_layer/leaky_reluMaximum$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
LGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  �=*
dtype0
�
TGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
seed2 
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/subSubJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
�
FGenerator/last_layer/fully_connected/kernel/Initializer/random_uniformAddJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��
�
+Generator/last_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
��
�
2Generator/last_layer/fully_connected/kernel/AssignAssign+Generator/last_layer/fully_connected/kernelFGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
0Generator/last_layer/fully_connected/kernel/readIdentity+Generator/last_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��
�
KGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:�*
dtype0
�
AGenerator/last_layer/fully_connected/bias/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;Generator/last_layer/fully_connected/bias/Initializer/zerosFillKGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorAGenerator/last_layer/fully_connected/bias/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:�
�
)Generator/last_layer/fully_connected/bias
VariableV2*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
0Generator/last_layer/fully_connected/bias/AssignAssign)Generator/last_layer/fully_connected/bias;Generator/last_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
.Generator/last_layer/fully_connected/bias/readIdentity)Generator/last_layer/fully_connected/bias*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:�
�
+Generator/last_layer/fully_connected/MatMulMatMul Generator/third_layer/leaky_relu0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
,Generator/last_layer/fully_connected/BiasAddBiasAdd+Generator/last_layer/fully_connected/MatMul.Generator/last_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
OGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:�*
dtype0
�
EGenerator/last_layer/batch_normalization/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *  �?
�
?Generator/last_layer/batch_normalization/gamma/Initializer/onesFillOGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorEGenerator/last_layer/batch_normalization/gamma/Initializer/ones/Const*
_output_shapes	
:�*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
�
.Generator/last_layer/batch_normalization/gamma
VariableV2*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
5Generator/last_layer/batch_normalization/gamma/AssignAssign.Generator/last_layer/batch_normalization/gamma?Generator/last_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
3Generator/last_layer/batch_normalization/gamma/readIdentity.Generator/last_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:�
�
OGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:�*
dtype0*
_output_shapes
:
�
EGenerator/last_layer/batch_normalization/beta/Initializer/zeros/ConstConst*
_output_shapes
: *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0
�
?Generator/last_layer/batch_normalization/beta/Initializer/zerosFillOGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorEGenerator/last_layer/batch_normalization/beta/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:�
�
-Generator/last_layer/batch_normalization/beta
VariableV2*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
4Generator/last_layer/batch_normalization/beta/AssignAssign-Generator/last_layer/batch_normalization/beta?Generator/last_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
2Generator/last_layer/batch_normalization/beta/readIdentity-Generator/last_layer/batch_normalization/beta*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:�*
T0
�
VGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB:�*
dtype0*
_output_shapes
:
�
LGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/ConstConst*
_output_shapes
: *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB
 *    *
dtype0
�
FGenerator/last_layer/batch_normalization/moving_mean/Initializer/zerosFillVGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*

index_type0*
_output_shapes	
:�
�
4Generator/last_layer/batch_normalization/moving_mean
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
	container 
�
;Generator/last_layer/batch_normalization/moving_mean/AssignAssign4Generator/last_layer/batch_normalization/moving_meanFGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:�
�
9Generator/last_layer/batch_normalization/moving_mean/readIdentity4Generator/last_layer/batch_normalization/moving_mean*
_output_shapes	
:�*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean
�
YGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB:�*
dtype0*
_output_shapes
:
�
OGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/ConstConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
IGenerator/last_layer/batch_normalization/moving_variance/Initializer/onesFillYGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorOGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/Const*
_output_shapes	
:�*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*

index_type0
�
8Generator/last_layer/batch_normalization/moving_variance
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance
�
?Generator/last_layer/batch_normalization/moving_variance/AssignAssign8Generator/last_layer/batch_normalization/moving_varianceIGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:�
�
=Generator/last_layer/batch_normalization/moving_variance/readIdentity8Generator/last_layer/batch_normalization/moving_variance*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
_output_shapes	
:�
}
8Generator/last_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
6Generator/last_layer/batch_normalization/batchnorm/addAdd=Generator/last_layer/batch_normalization/moving_variance/read8Generator/last_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:�*
T0
�
8Generator/last_layer/batch_normalization/batchnorm/RsqrtRsqrt6Generator/last_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:�
�
6Generator/last_layer/batch_normalization/batchnorm/mulMul8Generator/last_layer/batch_normalization/batchnorm/Rsqrt3Generator/last_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
�
8Generator/last_layer/batch_normalization/batchnorm/mul_1Mul,Generator/last_layer/fully_connected/BiasAdd6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
8Generator/last_layer/batch_normalization/batchnorm/mul_2Mul9Generator/last_layer/batch_normalization/moving_mean/read6Generator/last_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:�*
T0
�
6Generator/last_layer/batch_normalization/batchnorm/subSub2Generator/last_layer/batch_normalization/beta/read8Generator/last_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
8Generator/last_layer/batch_normalization/batchnorm/add_1Add8Generator/last_layer/batch_normalization/batchnorm/mul_16Generator/last_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:����������
j
%Generator/last_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
�
#Generator/last_layer/leaky_relu/mulMul%Generator/last_layer/leaky_relu/alpha8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Generator/last_layer/leaky_reluMaximum#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
<Generator/fake_image/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
:Generator/fake_image/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *z�k�*
dtype0*
_output_shapes
: 
�
:Generator/fake_image/kernel/Initializer/random_uniform/maxConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *z�k=*
dtype0*
_output_shapes
: 
�
DGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniformRandomUniform<Generator/fake_image/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
��*

seed *
T0*.
_class$
" loc:@Generator/fake_image/kernel
�
:Generator/fake_image/kernel/Initializer/random_uniform/subSub:Generator/fake_image/kernel/Initializer/random_uniform/max:Generator/fake_image/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*.
_class$
" loc:@Generator/fake_image/kernel
�
:Generator/fake_image/kernel/Initializer/random_uniform/mulMulDGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniform:Generator/fake_image/kernel/Initializer/random_uniform/sub*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��*
T0
�
6Generator/fake_image/kernel/Initializer/random_uniformAdd:Generator/fake_image/kernel/Initializer/random_uniform/mul:Generator/fake_image/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��
�
Generator/fake_image/kernel
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel
�
"Generator/fake_image/kernel/AssignAssignGenerator/fake_image/kernel6Generator/fake_image/kernel/Initializer/random_uniform*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
��
�
 Generator/fake_image/kernel/readIdentityGenerator/fake_image/kernel*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��
�
+Generator/fake_image/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*,
_class"
 loc:@Generator/fake_image/bias*
valueB�*    
�
Generator/fake_image/bias
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *,
_class"
 loc:@Generator/fake_image/bias
�
 Generator/fake_image/bias/AssignAssignGenerator/fake_image/bias+Generator/fake_image/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias
�
Generator/fake_image/bias/readIdentityGenerator/fake_image/bias*
_output_shapes	
:�*
T0*,
_class"
 loc:@Generator/fake_image/bias
�
Generator/fake_image/MatMulMatMulGenerator/last_layer/leaky_relu Generator/fake_image/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
Generator/fake_image/BiasAddBiasAddGenerator/fake_image/MatMulGenerator/fake_image/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
r
Generator/fake_image/TanhTanhGenerator/fake_image/BiasAdd*(
_output_shapes
:����������*
T0
z
Discriminator/real_inPlaceholder*
dtype0*(
_output_shapes
:����������*
shape:����������
�
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY��*
dtype0*
_output_shapes
: 
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY�=*
dtype0*
_output_shapes
: 
�
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
seed2 
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
0Discriminator/first_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
��
�
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
@Discriminator/first_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    
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
5Discriminator/first_layer/fully_connected/bias/AssignAssign.Discriminator/first_layer/fully_connected/bias@Discriminator/first_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:�*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
0Discriminator/first_layer/fully_connected/MatMulMatMulDiscriminator/real_in5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
o
*Discriminator/first_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
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
dtype0* 
_output_shapes
:
��*

seed *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
seed2 
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
_output_shapes
: 
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
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
8Discriminator/second_layer/fully_connected/kernel/AssignAssign1Discriminator/second_layer/fully_connected/kernelLDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container 
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
8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
: 
�
8Discriminator/prob/kernel/Initializer/random_uniform/mulMulBDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniform8Discriminator/prob/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
4Discriminator/prob/kernel/Initializer/random_uniformAdd8Discriminator/prob/kernel/Initializer/random_uniform/mul8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
Discriminator/prob/kernel
VariableV2*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel
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
Discriminator/prob/bias/AssignAssignDiscriminator/prob/bias)Discriminator/prob/bias/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
�
Discriminator/prob/bias/readIdentityDiscriminator/prob/bias*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
�
Discriminator/prob/MatMulMatMul%Discriminator/second_layer/leaky_reluDiscriminator/prob/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
Discriminator/prob/BiasAddBiasAddDiscriminator/prob/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
s
Discriminator/prob/SigmoidSigmoidDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
2Discriminator/first_layer_1/fully_connected/MatMulMatMulGenerator/fake_image/Tanh5Discriminator/first_layer/fully_connected/kernel/read*
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
,Discriminator/first_layer_1/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
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
Discriminator/prob_1/MatMulMatMul'Discriminator/second_layer_1/leaky_reluDiscriminator/prob/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Discriminator/prob_1/BiasAddBiasAddDiscriminator/prob_1/MatMulDiscriminator/prob/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
w
Discriminator/prob_1/SigmoidSigmoidDiscriminator/prob_1/BiasAdd*'
_output_shapes
:���������*
T0
X
LogLogDiscriminator/prob/Sigmoid*
T0*'
_output_shapes
:���������
J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
a
subSubsub/xDiscriminator/prob_1/Sigmoid*
T0*'
_output_shapes
:���������
C
Log_1Logsub*
T0*'
_output_shapes
:���������
H
addAddLogLog_1*
T0*'
_output_shapes
:���������
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
V
MeanMeanaddConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
1
NegNegMean*
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
discriminator_lossHistogramSummarydiscriminator_loss/tagNeg*
T0*
_output_shapes
: 
L
sub_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
e
sub_1Subsub_1/xDiscriminator/prob_1/Sigmoid*'
_output_shapes
:���������*
T0
E
Log_2Logsub_1*
T0*'
_output_shapes
:���������
X
Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
\
Mean_1MeanLog_2Const_1*
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
generator_lossHistogramSummarygenerator_loss/tagMean_1*
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
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
\
gradients/Mean_grad/ShapeShapeadd*
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
^
gradients/Mean_grad/Shape_1Shapeadd*
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
[
gradients/add_grad/ShapeShapeLog*
T0*
out_type0*
_output_shapes
:
_
gradients/add_grad/Shape_1ShapeLog_1*
T0*
out_type0*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Mean_grad/truediv(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sumgradients/Mean_grad/truediv*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
�
gradients/Log_grad/Reciprocal
ReciprocalDiscriminator/prob/Sigmoid,^gradients/add_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
gradients/Log_grad/mulMul+gradients/add_grad/tuple/control_dependencygradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
gradients/Log_1_grad/Reciprocal
Reciprocalsub.^gradients/add_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
gradients/Log_1_grad/mulMul-gradients/add_grad/tuple/control_dependency_1gradients/Log_1_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
5gradients/Discriminator/prob/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/prob/Sigmoidgradients/Log_grad/mul*
T0*'
_output_shapes
:���������
[
gradients/sub_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
v
gradients/sub_grad/Shape_1ShapeDiscriminator/prob_1/Sigmoid*
T0*
out_type0*
_output_shapes
:
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSumgradients/Log_1_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
gradients/sub_grad/Sum_1Sumgradients/Log_1_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
: 
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*'
_output_shapes
:���������
�
5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/Discriminator/prob/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp6^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad6^gradients/Discriminator/prob/Sigmoid_grad/SigmoidGrad
�
Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/Discriminator/prob/Sigmoid_grad/SigmoidGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������
�
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad
�
7gradients/Discriminator/prob_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/prob_1/Sigmoid-gradients/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
1gradients/Discriminator/prob/MatMul_grad/MatMul_1MatMul%Discriminator/second_layer/leaky_reluBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	�*
transpose_a(
�
9gradients/Discriminator/prob/MatMul_grad/tuple/group_depsNoOp0^gradients/Discriminator/prob/MatMul_grad/MatMul2^gradients/Discriminator/prob/MatMul_grad/MatMul_1
�
Agradients/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity/gradients/Discriminator/prob/MatMul_grad/MatMul:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*B
_class8
64loc:@gradients/Discriminator/prob/MatMul_grad/MatMul
�
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp8^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad8^gradients/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad
�
Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentity7gradients/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������
�
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
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
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Jgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Discriminator/second_layer/leaky_relu_grad/Shape<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeReshape8gradients/Discriminator/second_layer/leaky_relu_grad/Sum:gradients/Discriminator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1Sum=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1Lgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Egradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_depsNoOp=^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape?^gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
�
Mgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity<gradients/Discriminator/second_layer/leaky_relu_grad/ReshapeF^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*O
_classE
CAloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape
�
Ogradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1F^gradients/Discriminator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
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
�
gradients/AddNAddNDgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
�
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
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
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Qgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeJ^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape
�
Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityBgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1J^gradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
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
Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
Lgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape:gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Ggradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOp?^gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeA^gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
Ogradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeH^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
gradients/AddN_1AddNCgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
N*
_output_shapes
:	�
�
gradients/AddN_2AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_2N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_2S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
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
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Rgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
gradients/AddN_3AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_3P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
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
<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Select@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqual9gradients/Discriminator/first_layer/leaky_relu_grad/zerosYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
7gradients/Discriminator/first_layer/leaky_relu_grad/SumSum:gradients/Discriminator/first_layer/leaky_relu_grad/SelectIgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeReshape7gradients/Discriminator/first_layer/leaky_relu_grad/Sum9gradients/Discriminator/first_layer/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
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
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
�
gradients/AddN_4AddN\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Fgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp>^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape@^gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
Ngradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeG^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape
�
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_5AddN[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
��
�
gradients/AddN_6AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
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
�
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Sum?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Qgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Cgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1Reshape?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
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
�
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/real_inYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul
�
Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_7AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_7O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_7T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
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
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
�
Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1S^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_8AddN[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
gradients/AddN_9AddNZgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
N* 
_output_shapes
:
��*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
beta1_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
�
beta1_power/readIdentitybeta1_power*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
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
VariableV2*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
WDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
MDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
GDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillWDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorMDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
5Discriminator/first_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
��
�
<Discriminator/first_layer/fully_connected/kernel/Adam/AssignAssign5Discriminator/first_layer/fully_connected/kernel/AdamGDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(
�
:Discriminator/first_layer/fully_connected/kernel/Adam/readIdentity5Discriminator/first_layer/fully_connected/kernel/Adam* 
_output_shapes
:
��*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
YDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
ODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillYDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0
�
7Discriminator/first_layer/fully_connected/kernel/Adam_1
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container 
�
>Discriminator/first_layer/fully_connected/kernel/Adam_1/AssignAssign7Discriminator/first_layer/fully_connected/kernel/Adam_1IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
<Discriminator/first_layer/fully_connected/kernel/Adam_1/readIdentity7Discriminator/first_layer/fully_connected/kernel/Adam_1*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
EDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3Discriminator/first_layer/fully_connected/bias/Adam
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
:Discriminator/first_layer/fully_connected/bias/Adam/AssignAssign3Discriminator/first_layer/fully_connected/bias/AdamEDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
8Discriminator/first_layer/fully_connected/bias/Adam/readIdentity3Discriminator/first_layer/fully_connected/bias/Adam*
_output_shapes	
:�*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5Discriminator/first_layer/fully_connected/bias/Adam_1
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
<Discriminator/first_layer/fully_connected/bias/Adam_1/AssignAssign5Discriminator/first_layer/fully_connected/bias/Adam_1GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
:Discriminator/first_layer/fully_connected/bias/Adam_1/readIdentity5Discriminator/first_layer/fully_connected/bias/Adam_1*
_output_shapes	
:�*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
XDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
NDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    
�
HDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillXDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorNDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
6Discriminator/second_layer/fully_connected/kernel/Adam
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
�
=Discriminator/second_layer/fully_connected/kernel/Adam/AssignAssign6Discriminator/second_layer/fully_connected/kernel/AdamHDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
;Discriminator/second_layer/fully_connected/kernel/Adam/readIdentity6Discriminator/second_layer/fully_connected/kernel/Adam*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
ZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0
�
PDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    
�
JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorPDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
8Discriminator/second_layer/fully_connected/kernel/Adam_1
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
?Discriminator/second_layer/fully_connected/kernel/Adam_1/AssignAssign8Discriminator/second_layer/fully_connected/kernel/Adam_1JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(
�
=Discriminator/second_layer/fully_connected/kernel/Adam_1/readIdentity8Discriminator/second_layer/fully_connected/kernel/Adam_1*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
FDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4Discriminator/second_layer/fully_connected/bias/Adam
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container 
�
;Discriminator/second_layer/fully_connected/bias/Adam/AssignAssign4Discriminator/second_layer/fully_connected/bias/AdamFDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
9Discriminator/second_layer/fully_connected/bias/Adam/readIdentity4Discriminator/second_layer/fully_connected/bias/Adam*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    
�
6Discriminator/second_layer/fully_connected/bias/Adam_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container 
�
=Discriminator/second_layer/fully_connected/bias/Adam_1/AssignAssign6Discriminator/second_layer/fully_connected/bias/Adam_1HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
;Discriminator/second_layer/fully_connected/bias/Adam_1/readIdentity6Discriminator/second_layer/fully_connected/bias/Adam_1*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
0Discriminator/prob/kernel/Adam/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
Discriminator/prob/kernel/Adam
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
%Discriminator/prob/kernel/Adam/AssignAssignDiscriminator/prob/kernel/Adam0Discriminator/prob/kernel/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�
�
#Discriminator/prob/kernel/Adam/readIdentityDiscriminator/prob/kernel/Adam*
_output_shapes
:	�*
T0*,
_class"
 loc:@Discriminator/prob/kernel
�
2Discriminator/prob/kernel/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
 Discriminator/prob/kernel/Adam_1
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
'Discriminator/prob/kernel/Adam_1/AssignAssign Discriminator/prob/kernel/Adam_12Discriminator/prob/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	�*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(
�
%Discriminator/prob/kernel/Adam_1/readIdentity Discriminator/prob/kernel/Adam_1*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
.Discriminator/prob/bias/Adam/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
�
Discriminator/prob/bias/Adam
VariableV2*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0
�
#Discriminator/prob/bias/Adam/AssignAssignDiscriminator/prob/bias/Adam.Discriminator/prob/bias/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
�
!Discriminator/prob/bias/Adam/readIdentityDiscriminator/prob/bias/Adam*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
�
0Discriminator/prob/bias/Adam_1/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
�
Discriminator/prob/bias/Adam_1
VariableV2**
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
%Discriminator/prob/bias/Adam_1/AssignAssignDiscriminator/prob/bias/Adam_10Discriminator/prob/bias/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:
�
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
 *�Q9
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
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
FAdam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam0Discriminator/first_layer/fully_connected/kernel5Discriminator/first_layer/fully_connected/kernel/Adam7Discriminator/first_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
use_locking( *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
DAdam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam.Discriminator/first_layer/fully_connected/bias3Discriminator/first_layer/fully_connected/bias/Adam5Discriminator/first_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
GAdam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernel6Discriminator/second_layer/fully_connected/kernel/Adam8Discriminator/second_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
�
EAdam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/bias4Discriminator/second_layer/fully_connected/bias/Adam6Discriminator/second_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
use_locking( *
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
/Adam/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernelDiscriminator/prob/kernel/Adam Discriminator/prob/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_nesterov( *
_output_shapes
:	�*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel
�
-Adam/update_Discriminator/prob/bias/ApplyAdam	ApplyAdamDiscriminator/prob/biasDiscriminator/prob/bias/AdamDiscriminator/prob/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN**
_class 
loc:@Discriminator/prob/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
Adam/mulMulbeta1_power/read
Adam/beta1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: *
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
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
 *  �?*
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
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
b
gradients_1/Mean_1_grad/ShapeShapeLog_2*
T0*
out_type0*
_output_shapes
:
�
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
d
gradients_1/Mean_1_grad/Shape_1ShapeLog_2*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_1_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_1/Mean_1_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
�
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
�
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*'
_output_shapes
:���������
�
!gradients_1/Log_2_grad/Reciprocal
Reciprocalsub_1 ^gradients_1/Mean_1_grad/truediv*
T0*'
_output_shapes
:���������
�
gradients_1/Log_2_grad/mulMulgradients_1/Mean_1_grad/truediv!gradients_1/Log_2_grad/Reciprocal*
T0*'
_output_shapes
:���������
_
gradients_1/sub_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
z
gradients_1/sub_1_grad/Shape_1ShapeDiscriminator/prob_1/Sigmoid*
T0*
out_type0*
_output_shapes
:
�
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/sub_1_grad/SumSumgradients_1/Log_2_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
gradients_1/sub_1_grad/Sum_1Sumgradients_1/Log_2_grad/mul.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
T0*
_output_shapes
:
�
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
�
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape
�
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*'
_output_shapes
:���������
�
9gradients_1/Discriminator/prob_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/prob_1/Sigmoid1gradients_1/sub_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients_1/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
>gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad:^gradients_1/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad
�
Fgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentity9gradients_1/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad
�
Hgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
3gradients_1/Discriminator/prob_1/MatMul_grad/MatMulMatMulFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
=gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul6^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1
�
Egradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Discriminator/prob_1/MatMul_grad/MatMul>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Ggradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*H
_class>
<:loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1
�
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
�
Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosFill@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:����������*
T0
�
Egradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Ngradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Agradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SumSum?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectNgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1SumAgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1Pgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Bgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Igradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOpA^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeC^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
Qgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeJ^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Sgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityBgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1J^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Rgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulRgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Tgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Fgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Mgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpE^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeG^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
�
Ugradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeN^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityFgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1N^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
gradients_1/AddNAddNSgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Qgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Vgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddNR^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddNW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
`gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityQgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Kgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Mgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Ugradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpL^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulN^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
�
]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityKgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulV^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
_gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityMgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1V^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zerosFill?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Dgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Mgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SumSum>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectMgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1Ogradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Agradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Hgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp@^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeB^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
Pgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeI^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Rgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityAgradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1I^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulQgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Sgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Egradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Lgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpD^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeF^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
�
Tgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeM^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityEgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1M^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_1AddNRgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
Pgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ugradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1Q^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1V^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
_gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradV^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
Jgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Lgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Tgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpK^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulM^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
�
\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityJgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulU^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityLgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1U^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
3gradients_1/Generator/fake_image/Tanh_grad/TanhGradTanhGradGenerator/fake_image/Tanh\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
>gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad4^gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
�
Fgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
�
Hgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
3gradients_1/Generator/fake_image/MatMul_grad/MatMulMatMulFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency Generator/fake_image/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1MatMulGenerator/last_layer/leaky_reluFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
=gradients_1/Generator/fake_image/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Generator/fake_image/MatMul_grad/MatMul6^gradients_1/Generator/fake_image/MatMul_grad/MatMul_1
�
Egradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/MatMul_grad/MatMul>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul
�
Ggradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul_1
�
6gradients_1/Generator/last_layer/leaky_relu_grad/ShapeShape#Generator/last_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2ShapeEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6gradients_1/Generator/last_layer/leaky_relu_grad/zerosFill8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Fgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Generator/last_layer/leaky_relu_grad/Shape8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients_1/Generator/last_layer/leaky_relu_grad/SelectSelect=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency6gradients_1/Generator/last_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Select=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqual6gradients_1/Generator/last_layer/leaky_relu_grad/zerosEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
4gradients_1/Generator/last_layer/leaky_relu_grad/SumSum7gradients_1/Generator/last_layer/leaky_relu_grad/SelectFgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Generator/last_layer/leaky_relu_grad/Sum6gradients_1/Generator/last_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Hgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_18gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Agradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape;^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1
�
Igradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeB^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*K
_classA
?=loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape
�
Kgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1B^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1
}
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Jgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulMulIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/SumSum8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulJgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Mul%Generator/last_layer/leaky_relu/alphaIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Egradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
�
Mgradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_2AddNKgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Generator/last_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_2_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_2agradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Generator/last_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Generator/last_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Generator/last_layer/fully_connected/BiasAddbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape
�
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg
�
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Igradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ngradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Vgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape
�
Xgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*\
_classR
PNloc:@gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Generator/last_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Generator/last_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:�
�
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
�
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*b
_classX
VTloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
Cgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Egradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/third_layer/leaky_reluVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Mgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1
�
Ugradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Wgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_3AddNdgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_33Generator/last_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_38Generator/last_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
7gradients_1/Generator/third_layer/leaky_relu_grad/ShapeShape$Generator/third_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7gradients_1/Generator/third_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Ggradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/third_layer/leaky_relu_grad/Shape9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients_1/Generator/third_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/third_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/third_layer/leaky_relu_grad/zerosUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
5gradients_1/Generator/third_layer/leaky_relu_grad/SumSum8gradients_1/Generator/third_layer/leaky_relu_grad/SelectGgradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/third_layer/leaky_relu_grad/Sum7gradients_1/Generator/third_layer/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Igradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_19gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Bgradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
�
Jgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Lgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
~
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Kgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/third_layer/leaky_relu/alphaJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Fgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
�
Ngradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*R
_classH
FDloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
�
gradients_1/AddN_4AddNLgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape9Generator/third_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_4`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4bgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
egradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape-Generator/third_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency7Generator/third_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumSumNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul-Generator/third_layer/fully_connected/BiasAddcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1bgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegNegegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpf^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1M^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg
�
agradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Jgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradcgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ogradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpd^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyK^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Wgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitycgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
Ygradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_17Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1:Generator/third_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:�
�
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulQ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul
�
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Dgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Fgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1MatMul!Generator/second_layer/leaky_reluWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Ngradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1
�
Vgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul
�
Xgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_5AddNegradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N
�
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_54Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_59Generator/third_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:�*
T0
�
Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpM^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
agradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�*
T0
�
8gradients_1/Generator/second_layer/leaky_relu_grad/ShapeShape%Generator/second_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
�
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2ShapeVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
8gradients_1/Generator/second_layer/leaky_relu_grad/zerosFill:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Hgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/Generator/second_layer/leaky_relu_grad/Shape:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
9gradients_1/Generator/second_layer/leaky_relu_grad/SelectSelect?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency8gradients_1/Generator/second_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Select?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqual8gradients_1/Generator/second_layer/leaky_relu_grad/zerosVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
6gradients_1/Generator/second_layer/leaky_relu_grad/SumSum9gradients_1/Generator/second_layer/leaky_relu_grad/SelectHgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeReshape6gradients_1/Generator/second_layer/leaky_relu_grad/Sum8gradients_1/Generator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1Sum;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Jgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1Reshape8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Cgradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_depsNoOp;^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape=^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1
�
Kgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeD^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Mgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1D^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Lgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulMulKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/SumSum:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulLgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeReshape:gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Mul'Generator/second_layer/leaky_relu/alphaKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Ngradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ggradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp?^gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeA^gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
�
Ogradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeH^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*Q
_classG
ECloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape
�
Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1H^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_6AddNMgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape:Generator/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_6agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6cgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape.Generator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
agradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mulagradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul.Generator/second_layer/fully_connected/BiasAdddgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1cgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/NegNegfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpg^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1N^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
�
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
�
Kgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGraddgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Pgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpe^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyL^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Xgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitydgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
Zgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*^
_classT
RPloc:@gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_18Generator/second_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:�*
T0
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Muldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1;Generator/second_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:�
�
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulR^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul
�
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Egradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulMatMulXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency2Generator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Ggradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/first_layer/leaky_reluXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Ogradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpF^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulH^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1
�
Wgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityEgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulP^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Ygradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityGgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1P^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_7AddNfgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_75Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_7:Generator/second_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpN^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
7gradients_1/Generator/first_layer/leaky_relu_grad/ShapeShape$Generator/first_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2ShapeWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7gradients_1/Generator/first_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Ggradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/first_layer/leaky_relu_grad/Shape9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients_1/Generator/first_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/first_layer/leaky_relu_grad/zerosWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
5gradients_1/Generator/first_layer/leaky_relu_grad/SumSum8gradients_1/Generator/first_layer/leaky_relu_grad/SelectGgradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/first_layer/leaky_relu_grad/Sum7gradients_1/Generator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Igradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_19gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Bgradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
�
Jgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Lgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
~
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Kgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/first_layer/leaky_relu/alphaJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Fgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
�
Ngradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_8AddNLgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Jgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ogradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_8K^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Wgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8P^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Ygradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Dgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/first_layer/fully_connected/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
�
Fgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise_inWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d�*
transpose_a(*
transpose_b( 
�
Ngradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Vgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*W
_classM
KIloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul
�
Xgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d�
�
beta1_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power_1
VariableV2*,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: *
use_locking(
|
beta1_power_1/readIdentitybeta1_power_1*
_output_shapes
: *
T0*,
_class"
 loc:@Generator/fake_image/bias
�
beta2_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: *
use_locking(
|
beta2_power_1/readIdentitybeta2_power_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
�
SGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
�
IGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
CGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
1Generator/first_layer/fully_connected/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	d�*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d�
�
8Generator/first_layer/fully_connected/kernel/Adam/AssignAssign1Generator/first_layer/fully_connected/kernel/AdamCGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
6Generator/first_layer/fully_connected/kernel/Adam/readIdentity1Generator/first_layer/fully_connected/kernel/Adam*
_output_shapes
:	d�*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
�
UGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
�
KGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
3Generator/first_layer/fully_connected/kernel/Adam_1
VariableV2*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�
�
:Generator/first_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/first_layer/fully_connected/kernel/Adam_1EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
8Generator/first_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/first_layer/fully_connected/kernel/Adam_1*
_output_shapes
:	d�*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
�
AGenerator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
/Generator/first_layer/fully_connected/bias/Adam
VariableV2*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
6Generator/first_layer/fully_connected/bias/Adam/AssignAssign/Generator/first_layer/fully_connected/bias/AdamAGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
4Generator/first_layer/fully_connected/bias/Adam/readIdentity/Generator/first_layer/fully_connected/bias/Adam*
_output_shapes	
:�*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
�
CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
1Generator/first_layer/fully_connected/bias/Adam_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container 
�
8Generator/first_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/first_layer/fully_connected/bias/Adam_1CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
6Generator/first_layer/fully_connected/bias/Adam_1/readIdentity1Generator/first_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:�
�
TGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
JGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
DGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillTGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorJGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0
�
2Generator/second_layer/fully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
��
�
9Generator/second_layer/fully_connected/kernel/Adam/AssignAssign2Generator/second_layer/fully_connected/kernel/AdamDGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
�
7Generator/second_layer/fully_connected/kernel/Adam/readIdentity2Generator/second_layer/fully_connected/kernel/Adam*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
VGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"�      
�
LGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    
�
FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillVGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
4Generator/second_layer/fully_connected/kernel/Adam_1
VariableV2*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
;Generator/second_layer/fully_connected/kernel/Adam_1/AssignAssign4Generator/second_layer/fully_connected/kernel/Adam_1FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
9Generator/second_layer/fully_connected/kernel/Adam_1/readIdentity4Generator/second_layer/fully_connected/kernel/Adam_1*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
BGenerator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
0Generator/second_layer/fully_connected/bias/Adam
VariableV2*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
7Generator/second_layer/fully_connected/bias/Adam/AssignAssign0Generator/second_layer/fully_connected/bias/AdamBGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
5Generator/second_layer/fully_connected/bias/Adam/readIdentity0Generator/second_layer/fully_connected/bias/Adam*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB�*    
�
2Generator/second_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:�
�
9Generator/second_layer/fully_connected/bias/Adam_1/AssignAssign2Generator/second_layer/fully_connected/bias/Adam_1DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
7Generator/second_layer/fully_connected/bias/Adam_1/readIdentity2Generator/second_layer/fully_connected/bias/Adam_1*
_output_shapes	
:�*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias
�
GGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5Generator/second_layer/batch_normalization/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:�
�
<Generator/second_layer/batch_normalization/gamma/Adam/AssignAssign5Generator/second_layer/batch_normalization/gamma/AdamGGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
:Generator/second_layer/batch_normalization/gamma/Adam/readIdentity5Generator/second_layer/batch_normalization/gamma/Adam*
_output_shapes	
:�*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
�
IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7Generator/second_layer/batch_normalization/gamma/Adam_1
VariableV2*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
>Generator/second_layer/batch_normalization/gamma/Adam_1/AssignAssign7Generator/second_layer/batch_normalization/gamma/Adam_1IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
�
<Generator/second_layer/batch_normalization/gamma/Adam_1/readIdentity7Generator/second_layer/batch_normalization/gamma/Adam_1*
_output_shapes	
:�*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
�
FGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4Generator/second_layer/batch_normalization/beta/Adam
VariableV2*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
;Generator/second_layer/batch_normalization/beta/Adam/AssignAssign4Generator/second_layer/batch_normalization/beta/AdamFGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
�
9Generator/second_layer/batch_normalization/beta/Adam/readIdentity4Generator/second_layer/batch_normalization/beta/Adam*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
6Generator/second_layer/batch_normalization/beta/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:�
�
=Generator/second_layer/batch_normalization/beta/Adam_1/AssignAssign6Generator/second_layer/batch_normalization/beta/Adam_1HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
;Generator/second_layer/batch_normalization/beta/Adam_1/readIdentity6Generator/second_layer/batch_normalization/beta/Adam_1*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
SGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
IGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
CGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
1Generator/third_layer/fully_connected/kernel/Adam
VariableV2*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
8Generator/third_layer/fully_connected/kernel/Adam/AssignAssign1Generator/third_layer/fully_connected/kernel/AdamCGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
6Generator/third_layer/fully_connected/kernel/Adam/readIdentity1Generator/third_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��
�
UGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
KGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    
�
EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
3Generator/third_layer/fully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
��
�
:Generator/third_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/third_layer/fully_connected/kernel/Adam_1EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
8Generator/third_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/third_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��
�
AGenerator/third_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
/Generator/third_layer/fully_connected/bias/Adam
VariableV2*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
6Generator/third_layer/fully_connected/bias/Adam/AssignAssign/Generator/third_layer/fully_connected/bias/AdamAGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
�
4Generator/third_layer/fully_connected/bias/Adam/readIdentity/Generator/third_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:�
�
CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
1Generator/third_layer/fully_connected/bias/Adam_1
VariableV2*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
8Generator/third_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/third_layer/fully_connected/bias/Adam_1CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
6Generator/third_layer/fully_connected/bias/Adam_1/readIdentity1Generator/third_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:�
�
FGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4Generator/third_layer/batch_normalization/gamma/Adam
VariableV2*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
;Generator/third_layer/batch_normalization/gamma/Adam/AssignAssign4Generator/third_layer/batch_normalization/gamma/AdamFGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
9Generator/third_layer/batch_normalization/gamma/Adam/readIdentity4Generator/third_layer/batch_normalization/gamma/Adam*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:�
�
HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
6Generator/third_layer/batch_normalization/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:�
�
=Generator/third_layer/batch_normalization/gamma/Adam_1/AssignAssign6Generator/third_layer/batch_normalization/gamma/Adam_1HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
;Generator/third_layer/batch_normalization/gamma/Adam_1/readIdentity6Generator/third_layer/batch_normalization/gamma/Adam_1*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:�
�
EGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3Generator/third_layer/batch_normalization/beta/Adam
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
�
:Generator/third_layer/batch_normalization/beta/Adam/AssignAssign3Generator/third_layer/batch_normalization/beta/AdamEGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
8Generator/third_layer/batch_normalization/beta/Adam/readIdentity3Generator/third_layer/batch_normalization/beta/Adam*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:�
�
GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5Generator/third_layer/batch_normalization/beta/Adam_1
VariableV2*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
<Generator/third_layer/batch_normalization/beta/Adam_1/AssignAssign5Generator/third_layer/batch_normalization/beta/Adam_1GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
:Generator/third_layer/batch_normalization/beta/Adam_1/readIdentity5Generator/third_layer/batch_normalization/beta/Adam_1*
_output_shapes	
:�*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
�
RGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
HGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    
�
BGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zerosFillRGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
0Generator/last_layer/fully_connected/kernel/Adam
VariableV2*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
7Generator/last_layer/fully_connected/kernel/Adam/AssignAssign0Generator/last_layer/fully_connected/kernel/AdamBGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
5Generator/last_layer/fully_connected/kernel/Adam/readIdentity0Generator/last_layer/fully_connected/kernel/Adam*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��
�
TGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      
�
JGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillTGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorJGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
2Generator/last_layer/fully_connected/kernel/Adam_1
VariableV2*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
9Generator/last_layer/fully_connected/kernel/Adam_1/AssignAssign2Generator/last_layer/fully_connected/kernel/Adam_1DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
7Generator/last_layer/fully_connected/kernel/Adam_1/readIdentity2Generator/last_layer/fully_connected/kernel/Adam_1*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��
�
PGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:�
�
FGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@Generator/last_layer/fully_connected/bias/Adam/Initializer/zerosFillPGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorFGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/Const*
_output_shapes	
:�*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0
�
.Generator/last_layer/fully_connected/bias/Adam
VariableV2*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
5Generator/last_layer/fully_connected/bias/Adam/AssignAssign.Generator/last_layer/fully_connected/bias/Adam@Generator/last_layer/fully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
�
3Generator/last_layer/fully_connected/bias/Adam/readIdentity.Generator/last_layer/fully_connected/bias/Adam*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:�
�
RGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
HGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zerosFillRGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:�
�
0Generator/last_layer/fully_connected/bias/Adam_1
VariableV2*
_output_shapes	
:�*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:�*
dtype0
�
7Generator/last_layer/fully_connected/bias/Adam_1/AssignAssign0Generator/last_layer/fully_connected/bias/Adam_1BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
�
5Generator/last_layer/fully_connected/bias/Adam_1/readIdentity0Generator/last_layer/fully_connected/bias/Adam_1*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:�
�
UGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:�*
dtype0
�
KGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
�
EGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zerosFillUGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorKGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/Const*
_output_shapes	
:�*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
�
3Generator/last_layer/batch_normalization/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:�
�
:Generator/last_layer/batch_normalization/gamma/Adam/AssignAssign3Generator/last_layer/batch_normalization/gamma/AdamEGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
8Generator/last_layer/batch_normalization/gamma/Adam/readIdentity3Generator/last_layer/batch_normalization/gamma/Adam*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:�*
T0
�
WGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:�*
dtype0*
_output_shapes
:
�
MGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
�
GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zerosFillWGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorMGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/Const*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:�*
T0
�
5Generator/last_layer/batch_normalization/gamma/Adam_1
VariableV2*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
<Generator/last_layer/batch_normalization/gamma/Adam_1/AssignAssign5Generator/last_layer/batch_normalization/gamma/Adam_1GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma
�
:Generator/last_layer/batch_normalization/gamma/Adam_1/readIdentity5Generator/last_layer/batch_normalization/gamma/Adam_1*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:�*
T0
�
TGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:�
�
JGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
DGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zerosFillTGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorJGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/Const*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:�*
T0
�
2Generator/last_layer/batch_normalization/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:�
�
9Generator/last_layer/batch_normalization/beta/Adam/AssignAssign2Generator/last_layer/batch_normalization/beta/AdamDGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
7Generator/last_layer/batch_normalization/beta/Adam/readIdentity2Generator/last_layer/batch_normalization/beta/Adam*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:�
�
VGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:�*
dtype0*
_output_shapes
:
�
LGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zerosFillVGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:�
�
4Generator/last_layer/batch_normalization/beta/Adam_1
VariableV2*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
;Generator/last_layer/batch_normalization/beta/Adam_1/AssignAssign4Generator/last_layer/batch_normalization/beta/Adam_1FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
9Generator/last_layer/batch_normalization/beta/Adam_1/readIdentity4Generator/last_layer/batch_normalization/beta/Adam_1*
_output_shapes	
:�*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
�
BGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     
�
8Generator/fake_image/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
2Generator/fake_image/kernel/Adam/Initializer/zerosFillBGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensor8Generator/fake_image/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0
�
 Generator/fake_image/kernel/Adam
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel
�
'Generator/fake_image/kernel/Adam/AssignAssign Generator/fake_image/kernel/Adam2Generator/fake_image/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
��
�
%Generator/fake_image/kernel/Adam/readIdentity Generator/fake_image/kernel/Adam* 
_output_shapes
:
��*
T0*.
_class$
" loc:@Generator/fake_image/kernel
�
DGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
:Generator/fake_image/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
4Generator/fake_image/kernel/Adam_1/Initializer/zerosFillDGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensor:Generator/fake_image/kernel/Adam_1/Initializer/zeros/Const*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
"Generator/fake_image/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
��
�
)Generator/fake_image/kernel/Adam_1/AssignAssign"Generator/fake_image/kernel/Adam_14Generator/fake_image/kernel/Adam_1/Initializer/zeros*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
'Generator/fake_image/kernel/Adam_1/readIdentity"Generator/fake_image/kernel/Adam_1*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��*
T0
�
0Generator/fake_image/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*,
_class"
 loc:@Generator/fake_image/bias*
valueB�*    
�
Generator/fake_image/bias/Adam
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
%Generator/fake_image/bias/Adam/AssignAssignGenerator/fake_image/bias/Adam0Generator/fake_image/bias/Adam/Initializer/zeros*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
#Generator/fake_image/bias/Adam/readIdentityGenerator/fake_image/bias/Adam*
_output_shapes	
:�*
T0*,
_class"
 loc:@Generator/fake_image/bias
�
2Generator/fake_image/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*,
_class"
 loc:@Generator/fake_image/bias*
valueB�*    *
dtype0
�
 Generator/fake_image/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:�
�
'Generator/fake_image/bias/Adam_1/AssignAssign Generator/fake_image/bias/Adam_12Generator/fake_image/bias/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
%Generator/fake_image/bias/Adam_1/readIdentity Generator/fake_image/bias/Adam_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:�
Y
Adam_1/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *�Q9
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
 *w�?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
DAdam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/first_layer/fully_connected/kernel1Generator/first_layer/fully_connected/kernel/Adam3Generator/first_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
use_nesterov( *
_output_shapes
:	d�
�
BAdam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/first_layer/fully_connected/bias/Generator/first_layer/fully_connected/bias/Adam1Generator/first_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
EAdam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam-Generator/second_layer/fully_connected/kernel2Generator/second_layer/fully_connected/kernel/Adam4Generator/second_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
�
CAdam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam+Generator/second_layer/fully_connected/bias0Generator/second_layer/fully_connected/bias/Adam2Generator/second_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
HAdam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam0Generator/second_layer/batch_normalization/gamma5Generator/second_layer/batch_normalization/gamma/Adam7Generator/second_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
GAdam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam/Generator/second_layer/batch_normalization/beta4Generator/second_layer/batch_normalization/beta/Adam6Generator/second_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:�
�
DAdam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/third_layer/fully_connected/kernel1Generator/third_layer/fully_connected/kernel/Adam3Generator/third_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
BAdam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/third_layer/fully_connected/bias/Generator/third_layer/fully_connected/bias/Adam1Generator/third_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:�*
use_locking( *
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
use_nesterov( 
�
GAdam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam/Generator/third_layer/batch_normalization/gamma4Generator/third_layer/batch_normalization/gamma/Adam6Generator/third_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�
�
FAdam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdam	ApplyAdam.Generator/third_layer/batch_normalization/beta3Generator/third_layer/batch_normalization/beta/Adam5Generator/third_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
�
CAdam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Generator/last_layer/fully_connected/kernel0Generator/last_layer/fully_connected/kernel/Adam2Generator/last_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
�
AAdam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Generator/last_layer/fully_connected/bias.Generator/last_layer/fully_connected/bias/Adam0Generator/last_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:�*
use_locking( *
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
use_nesterov( 
�
FAdam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Generator/last_layer/batch_normalization/gamma3Generator/last_layer/batch_normalization/gamma/Adam5Generator/last_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�
�
EAdam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Generator/last_layer/batch_normalization/beta2Generator/last_layer/batch_normalization/beta/Adam4Generator/last_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:�
�
3Adam_1/update_Generator/fake_image/kernel/ApplyAdam	ApplyAdamGenerator/fake_image/kernel Generator/fake_image/kernel/Adam"Generator/fake_image/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
use_locking( *
T0*.
_class$
" loc:@Generator/fake_image/kernel*
use_nesterov( 
�
1Adam_1/update_Generator/fake_image/bias/ApplyAdam	ApplyAdamGenerator/fake_image/biasGenerator/fake_image/bias/Adam Generator/fake_image/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1*,
_class"
 loc:@Generator/fake_image/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
�	
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta22^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*,
_class"
 loc:@Generator/fake_image/bias
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: "<8ɜ�     �A$	�c>����AJ��
��
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
,
Log
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
shared_namestring �*1.12.02v1.12.0-0-ga6d8ffae09��
u
Generator/noise_inPlaceholder*
shape:���������d*
dtype0*'
_output_shapes
:���������d
�
MGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   �   
�
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&�*
dtype0*
_output_shapes
: 
�
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&>
�
UGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes
:	d�*

seed *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
�
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
: 
�
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
GGenerator/first_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
,Generator/first_layer/fully_connected/kernel
VariableV2*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name 
�
3Generator/first_layer/fully_connected/kernel/AssignAssign,Generator/first_layer/fully_connected/kernelGGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
1Generator/first_layer/fully_connected/kernel/readIdentity,Generator/first_layer/fully_connected/kernel*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
<Generator/first_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
*Generator/first_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:�
�
1Generator/first_layer/fully_connected/bias/AssignAssign*Generator/first_layer/fully_connected/bias<Generator/first_layer/fully_connected/bias/Initializer/zeros*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
/Generator/first_layer/fully_connected/bias/readIdentity*Generator/first_layer/fully_connected/bias*
_output_shapes	
:�*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
�
,Generator/first_layer/fully_connected/MatMulMatMulGenerator/noise_in1Generator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
-Generator/first_layer/fully_connected/BiasAddBiasAdd,Generator/first_layer/fully_connected/MatMul/Generator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
k
&Generator/first_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
�
$Generator/first_layer/leaky_relu/mulMul&Generator/first_layer/leaky_relu/alpha-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
 Generator/first_layer/leaky_reluMaximum$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
NGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   �*
dtype0*
_output_shapes
: 
�
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
�
VGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformNGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
seed2 
�
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
_output_shapes
: 
�
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulVGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
HGenerator/second_layer/fully_connected/kernel/Initializer/random_uniformAddLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
-Generator/second_layer/fully_connected/kernel
VariableV2*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
4Generator/second_layer/fully_connected/kernel/AssignAssign-Generator/second_layer/fully_connected/kernelHGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
2Generator/second_layer/fully_connected/kernel/readIdentity-Generator/second_layer/fully_connected/kernel*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
=Generator/second_layer/fully_connected/bias/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
+Generator/second_layer/fully_connected/bias
VariableV2*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
2Generator/second_layer/fully_connected/bias/AssignAssign+Generator/second_layer/fully_connected/bias=Generator/second_layer/fully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias
�
0Generator/second_layer/fully_connected/bias/readIdentity+Generator/second_layer/fully_connected/bias*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
-Generator/second_layer/fully_connected/MatMulMatMul Generator/first_layer/leaky_relu2Generator/second_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
.Generator/second_layer/fully_connected/BiasAddBiasAdd-Generator/second_layer/fully_connected/MatMul0Generator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
AGenerator/second_layer/batch_normalization/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:�*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB�*  �?
�
0Generator/second_layer/batch_normalization/gamma
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:�
�
7Generator/second_layer/batch_normalization/gamma/AssignAssign0Generator/second_layer/batch_normalization/gammaAGenerator/second_layer/batch_normalization/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
�
5Generator/second_layer/batch_normalization/gamma/readIdentity0Generator/second_layer/batch_normalization/gamma*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
AGenerator/second_layer/batch_normalization/beta/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
/Generator/second_layer/batch_normalization/beta
VariableV2*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
6Generator/second_layer/batch_normalization/beta/AssignAssign/Generator/second_layer/batch_normalization/betaAGenerator/second_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
4Generator/second_layer/batch_normalization/beta/readIdentity/Generator/second_layer/batch_normalization/beta*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
HGenerator/second_layer/batch_normalization/moving_mean/Initializer/zerosConst*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
6Generator/second_layer/batch_normalization/moving_mean
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
	container 
�
=Generator/second_layer/batch_normalization/moving_mean/AssignAssign6Generator/second_layer/batch_normalization/moving_meanHGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:�
�
;Generator/second_layer/batch_normalization/moving_mean/readIdentity6Generator/second_layer/batch_normalization/moving_mean*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
_output_shapes	
:�
�
KGenerator/second_layer/batch_normalization/moving_variance/Initializer/onesConst*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
:Generator/second_layer/batch_normalization/moving_variance
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance
�
AGenerator/second_layer/batch_normalization/moving_variance/AssignAssign:Generator/second_layer/batch_normalization/moving_varianceKGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance
�
?Generator/second_layer/batch_normalization/moving_variance/readIdentity:Generator/second_layer/batch_normalization/moving_variance*
_output_shapes	
:�*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance

:Generator/second_layer/batch_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
8Generator/second_layer/batch_normalization/batchnorm/addAdd?Generator/second_layer/batch_normalization/moving_variance/read:Generator/second_layer/batch_normalization/batchnorm/add/y*
_output_shapes	
:�*
T0
�
:Generator/second_layer/batch_normalization/batchnorm/RsqrtRsqrt8Generator/second_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:�
�
8Generator/second_layer/batch_normalization/batchnorm/mulMul:Generator/second_layer/batch_normalization/batchnorm/Rsqrt5Generator/second_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
�
:Generator/second_layer/batch_normalization/batchnorm/mul_1Mul.Generator/second_layer/fully_connected/BiasAdd8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
:Generator/second_layer/batch_normalization/batchnorm/mul_2Mul;Generator/second_layer/batch_normalization/moving_mean/read8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
8Generator/second_layer/batch_normalization/batchnorm/subSub4Generator/second_layer/batch_normalization/beta/read:Generator/second_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
:Generator/second_layer/batch_normalization/batchnorm/add_1Add:Generator/second_layer/batch_normalization/batchnorm/mul_18Generator/second_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:����������
l
'Generator/second_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
%Generator/second_layer/leaky_relu/mulMul'Generator/second_layer/leaky_relu/alpha:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
!Generator/second_layer/leaky_reluMaximum%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
MGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/minConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *��=*
dtype0*
_output_shapes
: 
�
UGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
��*

seed *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
�
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
�
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��
�
GGenerator/third_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
�
,Generator/third_layer/fully_connected/kernel
VariableV2*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
3Generator/third_layer/fully_connected/kernel/AssignAssign,Generator/third_layer/fully_connected/kernelGGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
1Generator/third_layer/fully_connected/kernel/readIdentity,Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
�
<Generator/third_layer/fully_connected/bias/Initializer/zerosConst*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
*Generator/third_layer/fully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:�
�
1Generator/third_layer/fully_connected/bias/AssignAssign*Generator/third_layer/fully_connected/bias<Generator/third_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
/Generator/third_layer/fully_connected/bias/readIdentity*Generator/third_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:�
�
,Generator/third_layer/fully_connected/MatMulMatMul!Generator/second_layer/leaky_relu1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
-Generator/third_layer/fully_connected/BiasAddBiasAdd,Generator/third_layer/fully_connected/MatMul/Generator/third_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
�
@Generator/third_layer/batch_normalization/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:�*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB�*  �?
�
/Generator/third_layer/batch_normalization/gamma
VariableV2*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
6Generator/third_layer/batch_normalization/gamma/AssignAssign/Generator/third_layer/batch_normalization/gamma@Generator/third_layer/batch_normalization/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma
�
4Generator/third_layer/batch_normalization/gamma/readIdentity/Generator/third_layer/batch_normalization/gamma*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:�
�
@Generator/third_layer/batch_normalization/beta/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
.Generator/third_layer/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:�
�
5Generator/third_layer/batch_normalization/beta/AssignAssign.Generator/third_layer/batch_normalization/beta@Generator/third_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
3Generator/third_layer/batch_normalization/beta/readIdentity.Generator/third_layer/batch_normalization/beta*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:�
�
GGenerator/third_layer/batch_normalization/moving_mean/Initializer/zerosConst*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5Generator/third_layer/batch_normalization/moving_mean
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
	container *
shape:�
�
<Generator/third_layer/batch_normalization/moving_mean/AssignAssign5Generator/third_layer/batch_normalization/moving_meanGGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:�
�
:Generator/third_layer/batch_normalization/moving_mean/readIdentity5Generator/third_layer/batch_normalization/moving_mean*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
_output_shapes	
:�
�
JGenerator/third_layer/batch_normalization/moving_variance/Initializer/onesConst*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
9Generator/third_layer/batch_normalization/moving_variance
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance
�
@Generator/third_layer/batch_normalization/moving_variance/AssignAssign9Generator/third_layer/batch_normalization/moving_varianceJGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:�
�
>Generator/third_layer/batch_normalization/moving_variance/readIdentity9Generator/third_layer/batch_normalization/moving_variance*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
_output_shapes	
:�
~
9Generator/third_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
7Generator/third_layer/batch_normalization/batchnorm/addAdd>Generator/third_layer/batch_normalization/moving_variance/read9Generator/third_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:�
�
9Generator/third_layer/batch_normalization/batchnorm/RsqrtRsqrt7Generator/third_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:�
�
7Generator/third_layer/batch_normalization/batchnorm/mulMul9Generator/third_layer/batch_normalization/batchnorm/Rsqrt4Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
9Generator/third_layer/batch_normalization/batchnorm/mul_1Mul-Generator/third_layer/fully_connected/BiasAdd7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
9Generator/third_layer/batch_normalization/batchnorm/mul_2Mul:Generator/third_layer/batch_normalization/moving_mean/read7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
7Generator/third_layer/batch_normalization/batchnorm/subSub3Generator/third_layer/batch_normalization/beta/read9Generator/third_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
9Generator/third_layer/batch_normalization/batchnorm/add_1Add9Generator/third_layer/batch_normalization/batchnorm/mul_17Generator/third_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:����������
k
&Generator/third_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
$Generator/third_layer/leaky_relu/mulMul&Generator/third_layer/leaky_relu/alpha9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
 Generator/third_layer/leaky_reluMaximum$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
LGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/minConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *  �=*
dtype0*
_output_shapes
: 
�
TGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
seed2 
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/subSubJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��
�
FGenerator/last_layer/fully_connected/kernel/Initializer/random_uniformAddJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
�
+Generator/last_layer/fully_connected/kernel
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
�
2Generator/last_layer/fully_connected/kernel/AssignAssign+Generator/last_layer/fully_connected/kernelFGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
�
0Generator/last_layer/fully_connected/kernel/readIdentity+Generator/last_layer/fully_connected/kernel*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��
�
KGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
AGenerator/last_layer/fully_connected/bias/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;Generator/last_layer/fully_connected/bias/Initializer/zerosFillKGenerator/last_layer/fully_connected/bias/Initializer/zeros/shape_as_tensorAGenerator/last_layer/fully_connected/bias/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:�
�
)Generator/last_layer/fully_connected/bias
VariableV2*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
0Generator/last_layer/fully_connected/bias/AssignAssign)Generator/last_layer/fully_connected/bias;Generator/last_layer/fully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
�
.Generator/last_layer/fully_connected/bias/readIdentity)Generator/last_layer/fully_connected/bias*
_output_shapes	
:�*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
�
+Generator/last_layer/fully_connected/MatMulMatMul Generator/third_layer/leaky_relu0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
,Generator/last_layer/fully_connected/BiasAddBiasAdd+Generator/last_layer/fully_connected/MatMul.Generator/last_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
OGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:�*
dtype0*
_output_shapes
:
�
EGenerator/last_layer/batch_normalization/gamma/Initializer/ones/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
?Generator/last_layer/batch_normalization/gamma/Initializer/onesFillOGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorEGenerator/last_layer/batch_normalization/gamma/Initializer/ones/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:�
�
.Generator/last_layer/batch_normalization/gamma
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:�
�
5Generator/last_layer/batch_normalization/gamma/AssignAssign.Generator/last_layer/batch_normalization/gamma?Generator/last_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
3Generator/last_layer/batch_normalization/gamma/readIdentity.Generator/last_layer/batch_normalization/gamma*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:�
�
OGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:�
�
EGenerator/last_layer/batch_normalization/beta/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?Generator/last_layer/batch_normalization/beta/Initializer/zerosFillOGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorEGenerator/last_layer/batch_normalization/beta/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:�
�
-Generator/last_layer/batch_normalization/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:�
�
4Generator/last_layer/batch_normalization/beta/AssignAssign-Generator/last_layer/batch_normalization/beta?Generator/last_layer/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
2Generator/last_layer/batch_normalization/beta/readIdentity-Generator/last_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:�
�
VGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB:�
�
LGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB
 *    
�
FGenerator/last_layer/batch_normalization/moving_mean/Initializer/zerosFillVGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*

index_type0*
_output_shapes	
:�
�
4Generator/last_layer/batch_normalization/moving_mean
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
	container *
shape:�
�
;Generator/last_layer/batch_normalization/moving_mean/AssignAssign4Generator/last_layer/batch_normalization/moving_meanFGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean
�
9Generator/last_layer/batch_normalization/moving_mean/readIdentity4Generator/last_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
_output_shapes	
:�
�
YGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB:�
�
OGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/ConstConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
IGenerator/last_layer/batch_normalization/moving_variance/Initializer/onesFillYGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorOGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/Const*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*

index_type0*
_output_shapes	
:�
�
8Generator/last_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
	container *
shape:�
�
?Generator/last_layer/batch_normalization/moving_variance/AssignAssign8Generator/last_layer/batch_normalization/moving_varianceIGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance
�
=Generator/last_layer/batch_normalization/moving_variance/readIdentity8Generator/last_layer/batch_normalization/moving_variance*
_output_shapes	
:�*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance
}
8Generator/last_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
6Generator/last_layer/batch_normalization/batchnorm/addAdd=Generator/last_layer/batch_normalization/moving_variance/read8Generator/last_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:�
�
8Generator/last_layer/batch_normalization/batchnorm/RsqrtRsqrt6Generator/last_layer/batch_normalization/batchnorm/add*
T0*
_output_shapes	
:�
�
6Generator/last_layer/batch_normalization/batchnorm/mulMul8Generator/last_layer/batch_normalization/batchnorm/Rsqrt3Generator/last_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
8Generator/last_layer/batch_normalization/batchnorm/mul_1Mul,Generator/last_layer/fully_connected/BiasAdd6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
8Generator/last_layer/batch_normalization/batchnorm/mul_2Mul9Generator/last_layer/batch_normalization/moving_mean/read6Generator/last_layer/batch_normalization/batchnorm/mul*
_output_shapes	
:�*
T0
�
6Generator/last_layer/batch_normalization/batchnorm/subSub2Generator/last_layer/batch_normalization/beta/read8Generator/last_layer/batch_normalization/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
8Generator/last_layer/batch_normalization/batchnorm/add_1Add8Generator/last_layer/batch_normalization/batchnorm/mul_16Generator/last_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:����������
j
%Generator/last_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
�
#Generator/last_layer/leaky_relu/mulMul%Generator/last_layer/leaky_relu/alpha8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Generator/last_layer/leaky_reluMaximum#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
<Generator/fake_image/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
:Generator/fake_image/kernel/Initializer/random_uniform/minConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *z�k�*
dtype0*
_output_shapes
: 
�
:Generator/fake_image/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *z�k=
�
DGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniformRandomUniform<Generator/fake_image/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed *
T0*.
_class$
" loc:@Generator/fake_image/kernel*
seed2 
�
:Generator/fake_image/kernel/Initializer/random_uniform/subSub:Generator/fake_image/kernel/Initializer/random_uniform/max:Generator/fake_image/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
_output_shapes
: 
�
:Generator/fake_image/kernel/Initializer/random_uniform/mulMulDGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniform:Generator/fake_image/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*.
_class$
" loc:@Generator/fake_image/kernel
�
6Generator/fake_image/kernel/Initializer/random_uniformAdd:Generator/fake_image/kernel/Initializer/random_uniform/mul:Generator/fake_image/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��
�
Generator/fake_image/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
��
�
"Generator/fake_image/kernel/AssignAssignGenerator/fake_image/kernel6Generator/fake_image/kernel/Initializer/random_uniform*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
��
�
 Generator/fake_image/kernel/readIdentityGenerator/fake_image/kernel*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��
�
+Generator/fake_image/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*,
_class"
 loc:@Generator/fake_image/bias*
valueB�*    
�
Generator/fake_image/bias
VariableV2*,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
 Generator/fake_image/bias/AssignAssignGenerator/fake_image/bias+Generator/fake_image/bias/Initializer/zeros*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
Generator/fake_image/bias/readIdentityGenerator/fake_image/bias*
_output_shapes	
:�*
T0*,
_class"
 loc:@Generator/fake_image/bias
�
Generator/fake_image/MatMulMatMulGenerator/last_layer/leaky_relu Generator/fake_image/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
Generator/fake_image/BiasAddBiasAddGenerator/fake_image/MatMulGenerator/fake_image/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
r
Generator/fake_image/TanhTanhGenerator/fake_image/BiasAdd*(
_output_shapes
:����������*
T0
z
Discriminator/real_inPlaceholder*
dtype0*(
_output_shapes
:����������*
shape:����������
�
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/minConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY��*
dtype0*
_output_shapes
: 
�
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY�=*
dtype0*
_output_shapes
: 
�
YDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformQDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

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
_output_shapes
:
��
�
KDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniformAddODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
0Discriminator/first_layer/fully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
��
�
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
5Discriminator/first_layer/fully_connected/kernel/readIdentity0Discriminator/first_layer/fully_connected/kernel*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
��
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
VariableV2*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
5Discriminator/first_layer/fully_connected/bias/AssignAssign.Discriminator/first_layer/fully_connected/bias@Discriminator/first_layer/fully_connected/bias/Initializer/zeros*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
3Discriminator/first_layer/fully_connected/bias/readIdentity.Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:�*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
0Discriminator/first_layer/fully_connected/MatMulMatMulDiscriminator/real_in5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
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
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
$Discriminator/first_layer/leaky_reluMaximum(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
RDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *���
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *��=
�
ZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformRDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shape*

seed *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
��
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulZDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
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
8Discriminator/second_layer/fully_connected/kernel/AssignAssign1Discriminator/second_layer/fully_connected/kernelLDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
6Discriminator/second_layer/fully_connected/bias/AssignAssign/Discriminator/second_layer/fully_connected/biasADiscriminator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:�*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
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
%Discriminator/second_layer/leaky_reluMaximum)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
:Discriminator/prob/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*,
_class"
 loc:@Discriminator/prob/kernel*
valueB"      
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
BDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniformRandomUniform:Discriminator/prob/kernel/Initializer/random_uniform/shape*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
seed2 *
dtype0*
_output_shapes
:	�*

seed 
�
8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
: 
�
8Discriminator/prob/kernel/Initializer/random_uniform/mulMulBDiscriminator/prob/kernel/Initializer/random_uniform/RandomUniform8Discriminator/prob/kernel/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
4Discriminator/prob/kernel/Initializer/random_uniformAdd8Discriminator/prob/kernel/Initializer/random_uniform/mul8Discriminator/prob/kernel/Initializer/random_uniform/min*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
Discriminator/prob/kernel
VariableV2*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
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
VariableV2**
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
Discriminator/prob/bias/AssignAssignDiscriminator/prob/bias)Discriminator/prob/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias
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
s
Discriminator/prob/SigmoidSigmoidDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
2Discriminator/first_layer_1/fully_connected/MatMulMatMulGenerator/fake_image/Tanh5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
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
'Discriminator/second_layer_1/leaky_reluMaximum+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
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
w
Discriminator/prob_1/SigmoidSigmoidDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
X
LogLogDiscriminator/prob/Sigmoid*'
_output_shapes
:���������*
T0
J
sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
a
subSubsub/xDiscriminator/prob_1/Sigmoid*
T0*'
_output_shapes
:���������
C
Log_1Logsub*
T0*'
_output_shapes
:���������
H
addAddLogLog_1*
T0*'
_output_shapes
:���������
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
V
MeanMeanaddConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
1
NegNegMean*
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
discriminator_lossHistogramSummarydiscriminator_loss/tagNeg*
T0*
_output_shapes
: 
L
sub_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
e
sub_1Subsub_1/xDiscriminator/prob_1/Sigmoid*'
_output_shapes
:���������*
T0
E
Log_2Logsub_1*'
_output_shapes
:���������*
T0
X
Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
\
Mean_1MeanLog_2Const_1*
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
generator_lossHistogramSummarygenerator_loss/tagMean_1*
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
 *  �?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
r
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
\
gradients/Mean_grad/ShapeShapeadd*
_output_shapes
:*
T0*
out_type0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
^
gradients/Mean_grad/Shape_1Shapeadd*
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
�
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
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
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
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:���������*
T0
[
gradients/add_grad/ShapeShapeLog*
_output_shapes
:*
T0*
out_type0
_
gradients/add_grad/Shape_1ShapeLog_1*
T0*
out_type0*
_output_shapes
:
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/Mean_grad/truediv(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sumgradients/Mean_grad/truediv*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:���������
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*'
_output_shapes
:���������
�
gradients/Log_grad/Reciprocal
ReciprocalDiscriminator/prob/Sigmoid,^gradients/add_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
gradients/Log_grad/mulMul+gradients/add_grad/tuple/control_dependencygradients/Log_grad/Reciprocal*'
_output_shapes
:���������*
T0
�
gradients/Log_1_grad/Reciprocal
Reciprocalsub.^gradients/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
gradients/Log_1_grad/mulMul-gradients/add_grad/tuple/control_dependency_1gradients/Log_1_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
5gradients/Discriminator/prob/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/prob/Sigmoidgradients/Log_grad/mul*'
_output_shapes
:���������*
T0
[
gradients/sub_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
v
gradients/sub_grad/Shape_1ShapeDiscriminator/prob_1/Sigmoid*
T0*
out_type0*
_output_shapes
:
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSumgradients/Log_1_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
gradients/sub_grad/Sum_1Sumgradients/Log_1_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
: 
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
�
5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/Discriminator/prob/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp6^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad6^gradients/Discriminator/prob/Sigmoid_grad/SigmoidGrad
�
Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/Discriminator/prob/Sigmoid_grad/SigmoidGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/Sigmoid_grad/SigmoidGrad
�
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
7gradients/Discriminator/prob_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/prob_1/Sigmoid-gradients/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
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
Agradients/Discriminator/prob/MatMul_grad/tuple/control_dependencyIdentity/gradients/Discriminator/prob/MatMul_grad/MatMul:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*B
_class8
64loc:@gradients/Discriminator/prob/MatMul_grad/MatMul
�
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp8^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad8^gradients/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad
�
Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentity7gradients/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������
�
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
�
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
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
:gradients/Discriminator/second_layer/leaky_relu_grad/zerosFill<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Jgradients/Discriminator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/Discriminator/second_layer/leaky_relu_grad/Shape<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
�
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1<^gradients/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*F
_class<
:8loc:@gradients/Discriminator/prob_1/MatMul_grad/MatMul_1
�
gradients/AddNAddNDgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
�
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1Shape2Discriminator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
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
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
@gradients/Discriminator/second_layer/leaky_relu/mul_grad/ReshapeReshape<gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosFill>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
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
:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape:gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
�
gradients/AddN_1AddNCgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
N*
_output_shapes
:	�
�
gradients/AddN_2AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_2N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_2S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1
�
\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityMgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradS^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
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
Pgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeBgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaOgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Rgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Dgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1Reshape@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
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
�
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Igradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul$Discriminator/first_layer/leaky_reluZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
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
gradients/AddN_3AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_3P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
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
9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1Sum<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Kgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Kgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
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
gradients/AddN_4AddN\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1Shape1Discriminator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Mgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape?gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMulLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency1Discriminator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
;gradients/Discriminator/first_layer/leaky_relu/mul_grad/SumSum;gradients/Discriminator/first_layer/leaky_relu/mul_grad/MulMgradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Mul*Discriminator/first_layer/leaky_relu/alphaLgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
�
;gradients/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
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
Kgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
Ngradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeG^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape
�
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_5AddN[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
��
�
gradients/AddN_6AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
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
Ogradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulOgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
�
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/real_inYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Pgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpG^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulI^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Xgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityFgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulQ^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Zgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityHgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1Q^gradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_7AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_7O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_7T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
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
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Rgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpI^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulK^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
�
Zgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulS^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1S^gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_8AddN[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
gradients/AddN_9AddNZgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
��
�
beta1_power/initial_valueConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power
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
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
beta1_power/readIdentitybeta1_power*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
WDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
MDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    
�
GDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillWDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorMDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
5Discriminator/first_layer/fully_connected/kernel/Adam
VariableV2*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
<Discriminator/first_layer/fully_connected/kernel/Adam/AssignAssign5Discriminator/first_layer/fully_connected/kernel/AdamGDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
:Discriminator/first_layer/fully_connected/kernel/Adam/readIdentity5Discriminator/first_layer/fully_connected/kernel/Adam*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
YDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
ODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillYDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
7Discriminator/first_layer/fully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
��
�
>Discriminator/first_layer/fully_connected/kernel/Adam_1/AssignAssign7Discriminator/first_layer/fully_connected/kernel/Adam_1IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
<Discriminator/first_layer/fully_connected/kernel/Adam_1/readIdentity7Discriminator/first_layer/fully_connected/kernel/Adam_1*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel* 
_output_shapes
:
��
�
EDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3Discriminator/first_layer/fully_connected/bias/Adam
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
:Discriminator/first_layer/fully_connected/bias/Adam/AssignAssign3Discriminator/first_layer/fully_connected/bias/AdamEDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
8Discriminator/first_layer/fully_connected/bias/Adam/readIdentity3Discriminator/first_layer/fully_connected/bias/Adam*
_output_shapes	
:�*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5Discriminator/first_layer/fully_connected/bias/Adam_1
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
<Discriminator/first_layer/fully_connected/bias/Adam_1/AssignAssign5Discriminator/first_layer/fully_connected/bias/Adam_1GDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
:Discriminator/first_layer/fully_connected/bias/Adam_1/readIdentity5Discriminator/first_layer/fully_connected/bias/Adam_1*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:�
�
XDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      
�
NDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
HDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillXDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorNDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0
�
6Discriminator/second_layer/fully_connected/kernel/Adam
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
�
=Discriminator/second_layer/fully_connected/kernel/Adam/AssignAssign6Discriminator/second_layer/fully_connected/kernel/AdamHDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
;Discriminator/second_layer/fully_connected/kernel/Adam/readIdentity6Discriminator/second_layer/fully_connected/kernel/Adam*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
ZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
PDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillZDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorPDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
8Discriminator/second_layer/fully_connected/kernel/Adam_1
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
?Discriminator/second_layer/fully_connected/kernel/Adam_1/AssignAssign8Discriminator/second_layer/fully_connected/kernel/Adam_1JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
=Discriminator/second_layer/fully_connected/kernel/Adam_1/readIdentity8Discriminator/second_layer/fully_connected/kernel/Adam_1*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
FDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    
�
4Discriminator/second_layer/fully_connected/bias/Adam
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
;Discriminator/second_layer/fully_connected/bias/Adam/AssignAssign4Discriminator/second_layer/fully_connected/bias/AdamFDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
9Discriminator/second_layer/fully_connected/bias/Adam/readIdentity4Discriminator/second_layer/fully_connected/bias/Adam*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
6Discriminator/second_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:�
�
=Discriminator/second_layer/fully_connected/bias/Adam_1/AssignAssign6Discriminator/second_layer/fully_connected/bias/Adam_1HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
;Discriminator/second_layer/fully_connected/bias/Adam_1/readIdentity6Discriminator/second_layer/fully_connected/bias/Adam_1*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
0Discriminator/prob/kernel/Adam/Initializer/zerosConst*
_output_shapes
:	�*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	�*    *
dtype0
�
Discriminator/prob/kernel/Adam
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
%Discriminator/prob/kernel/Adam/AssignAssignDiscriminator/prob/kernel/Adam0Discriminator/prob/kernel/Adam/Initializer/zeros*
_output_shapes
:	�*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(
�
#Discriminator/prob/kernel/Adam/readIdentityDiscriminator/prob/kernel/Adam*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
2Discriminator/prob/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:	�*,
_class"
 loc:@Discriminator/prob/kernel*
valueB	�*    
�
 Discriminator/prob/kernel/Adam_1
VariableV2*
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *,
_class"
 loc:@Discriminator/prob/kernel*
	container 
�
'Discriminator/prob/kernel/Adam_1/AssignAssign Discriminator/prob/kernel/Adam_12Discriminator/prob/kernel/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
%Discriminator/prob/kernel/Adam_1/readIdentity Discriminator/prob/kernel/Adam_1*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�
�
.Discriminator/prob/bias/Adam/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
�
Discriminator/prob/bias/Adam
VariableV2**
_class 
loc:@Discriminator/prob/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
#Discriminator/prob/bias/Adam/AssignAssignDiscriminator/prob/bias/Adam.Discriminator/prob/bias/Adam/Initializer/zeros*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
!Discriminator/prob/bias/Adam/readIdentityDiscriminator/prob/bias/Adam*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
�
0Discriminator/prob/bias/Adam_1/Initializer/zerosConst**
_class 
loc:@Discriminator/prob/bias*
valueB*    *
dtype0*
_output_shapes
:
�
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
�
%Discriminator/prob/bias/Adam_1/AssignAssignDiscriminator/prob/bias/Adam_10Discriminator/prob/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias
�
#Discriminator/prob/bias/Adam_1/readIdentityDiscriminator/prob/bias/Adam_1*
T0**
_class 
loc:@Discriminator/prob/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *�Q9*
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
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
FAdam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam0Discriminator/first_layer/fully_connected/kernel5Discriminator/first_layer/fully_connected/kernel/Adam7Discriminator/first_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
use_locking( *
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
DAdam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam.Discriminator/first_layer/fully_connected/bias3Discriminator/first_layer/fully_connected/bias/Adam5Discriminator/first_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
GAdam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernel6Discriminator/second_layer/fully_connected/kernel/Adam8Discriminator/second_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
EAdam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/bias4Discriminator/second_layer/fully_connected/bias/Adam6Discriminator/second_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
use_locking( *
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
/Adam/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernelDiscriminator/prob/kernel/Adam Discriminator/prob/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_nesterov( *
_output_shapes
:	�*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel
�
-Adam/update_Discriminator/prob/bias/ApplyAdam	ApplyAdamDiscriminator/prob/biasDiscriminator/prob/bias/AdamDiscriminator/prob/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0**
_class 
loc:@Discriminator/prob/bias
�
Adam/mulMulbeta1_power/read
Adam/beta1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes
: 
�
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
 *  �?*
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
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
b
gradients_1/Mean_1_grad/ShapeShapeLog_2*
T0*
out_type0*
_output_shapes
:
�
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
d
gradients_1/Mean_1_grad/Shape_1ShapeLog_2*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
c
!gradients_1/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
�
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
�
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*'
_output_shapes
:���������
�
!gradients_1/Log_2_grad/Reciprocal
Reciprocalsub_1 ^gradients_1/Mean_1_grad/truediv*'
_output_shapes
:���������*
T0
�
gradients_1/Log_2_grad/mulMulgradients_1/Mean_1_grad/truediv!gradients_1/Log_2_grad/Reciprocal*
T0*'
_output_shapes
:���������
_
gradients_1/sub_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
z
gradients_1/sub_1_grad/Shape_1ShapeDiscriminator/prob_1/Sigmoid*
_output_shapes
:*
T0*
out_type0
�
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/sub_1_grad/SumSumgradients_1/Log_2_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
gradients_1/sub_1_grad/Sum_1Sumgradients_1/Log_2_grad/mul.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
_output_shapes
:*
T0
�
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
�
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape
�
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*'
_output_shapes
:���������
�
9gradients_1/Discriminator/prob_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/prob_1/Sigmoid1gradients_1/sub_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients_1/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
>gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad:^gradients_1/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad
�
Fgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentity9gradients_1/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������
�
Hgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
3gradients_1/Discriminator/prob_1/MatMul_grad/MatMulMatMulFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
=gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul6^gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1
�
Egradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Discriminator/prob_1/MatMul_grad/MatMul>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Ggradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1>^gradients_1/Discriminator/prob_1/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosFill@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Egradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Ngradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Agradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SumSum?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectNgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape<gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1SumAgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1Pgradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Sum_1@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Igradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_depsNoOpA^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeC^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
Qgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/ReshapeJ^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Sgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityBgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1J^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Rgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ShapeDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulRgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Mul-Discriminator/second_layer_1/leaky_relu/alphaQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Tgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Fgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Mgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpE^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeG^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
�
Ugradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeN^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityFgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1N^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*Y
_classO
MKloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
�
gradients_1/AddNAddNSgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Qgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Vgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddNR^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddNW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
`gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityQgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*d
_classZ
XVloc:@gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
Kgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Mgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Ugradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpL^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulN^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
�
]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityKgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulV^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
_gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityMgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1V^gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ShapeShape*Discriminator/first_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zerosFill?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Dgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Mgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SumSum>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectMgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1Ogradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Agradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Hgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_depsNoOp@^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeB^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
Pgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependencyIdentity?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeI^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Rgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityAgradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1I^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1Shape3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulQgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeReshape?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Mul,Discriminator/first_layer_1/leaky_relu/alphaPgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Sgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Egradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Lgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpD^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeF^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
�
Tgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeM^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityEgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1M^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_1AddNRgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Pgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ugradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1Q^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1V^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
_gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradV^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
Jgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Lgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Tgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_depsNoOpK^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulM^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1
�
\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityJgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulU^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityLgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1U^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
3gradients_1/Generator/fake_image/Tanh_grad/TanhGradTanhGradGenerator/fake_image/Tanh\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
>gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad4^gradients_1/Generator/fake_image/Tanh_grad/TanhGrad
�
Fgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*(
_output_shapes
:����������
�
Hgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad?^gradients_1/Generator/fake_image/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
3gradients_1/Generator/fake_image/MatMul_grad/MatMulMatMulFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency Generator/fake_image/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1MatMulGenerator/last_layer/leaky_reluFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
=gradients_1/Generator/fake_image/MatMul_grad/tuple/group_depsNoOp4^gradients_1/Generator/fake_image/MatMul_grad/MatMul6^gradients_1/Generator/fake_image/MatMul_grad/MatMul_1
�
Egradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/Generator/fake_image/MatMul_grad/MatMul>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Ggradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/Generator/fake_image/MatMul_grad/MatMul_1>^gradients_1/Generator/fake_image/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/Generator/fake_image/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
6gradients_1/Generator/last_layer/leaky_relu_grad/ShapeShape#Generator/last_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2ShapeEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6gradients_1/Generator/last_layer/leaky_relu_grad/zerosFill8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualGreaterEqual#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Fgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/Generator/last_layer/leaky_relu_grad/Shape8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7gradients_1/Generator/last_layer/leaky_relu_grad/SelectSelect=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency6gradients_1/Generator/last_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Select=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqual6gradients_1/Generator/last_layer/leaky_relu_grad/zerosEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
4gradients_1/Generator/last_layer/leaky_relu_grad/SumSum7gradients_1/Generator/last_layer/leaky_relu_grad/SelectFgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeReshape4gradients_1/Generator/last_layer/leaky_relu_grad/Sum6gradients_1/Generator/last_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_1Sum9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Hgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1Reshape6gradients_1/Generator/last_layer/leaky_relu_grad/Sum_18gradients_1/Generator/last_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Agradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_depsNoOp9^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape;^gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1
�
Igradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeB^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*K
_classA
?=loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape
�
Kgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Identity:gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1B^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
}
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Jgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulMulIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/SumSum8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulJgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeReshape8gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Mul%Generator/last_layer/leaky_relu/alphaIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1Sum:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Mul_1Lgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Egradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_depsNoOp=^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape?^gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
�
Mgradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity<gradients_1/Generator/last_layer/leaky_relu/mul_grad/ReshapeF^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1F^gradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*Q
_classG
ECloc:@gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1
�
gradients_1/AddN_2AddNKgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Generator/last_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_2_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_2agradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Generator/last_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Generator/last_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Generator/last_layer/fully_connected/BiasAddbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpR^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeT^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegNegdgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpe^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1L^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg
�
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Igradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ngradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpc^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyJ^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Vgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitybgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
Xgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityIgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradO^gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_16Generator/last_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Generator/last_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:�
�
Zgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/MulP^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul
�
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Cgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Generator/last_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Egradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/third_layer/leaky_reluVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Mgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpD^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulF^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1
�
Ugradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Wgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_3AddNdgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*
_output_shapes	
:�*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_33Generator/last_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_38Generator/last_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
7gradients_1/Generator/third_layer/leaky_relu_grad/ShapeShape$Generator/third_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2ShapeUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7gradients_1/Generator/third_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/third_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Ggradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/third_layer/leaky_relu_grad/Shape9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients_1/Generator/third_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/third_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/third_layer/leaky_relu_grad/zerosUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
5gradients_1/Generator/third_layer/leaky_relu_grad/SumSum8gradients_1/Generator/third_layer/leaky_relu_grad/SelectGgradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/third_layer/leaky_relu_grad/Sum7gradients_1/Generator/third_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Igradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_19gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Bgradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
�
Jgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Lgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
~
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Kgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency9Generator/third_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
9gradients_1/Generator/third_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/third_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/third_layer/leaky_relu/alphaJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Fgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1
�
Ngradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Generator/third_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_4AddNLgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape9Generator/third_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_4`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4bgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
egradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape-Generator/third_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ShapeRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency7Generator/third_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumSumNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul`gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul-Generator/third_layer/fully_connected/BiasAddcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1bgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegNegegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpf^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1M^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg
�
agradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg
�
Jgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradcgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Ogradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpd^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyK^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Wgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitycgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
Ygradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_17Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1:Generator/third_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:�
�
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulQ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul
�
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Dgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/third_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Fgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1MatMul!Generator/second_layer/leaky_reluWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Ngradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1
�
Vgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul
�
Xgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_5AddNegradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_54Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_59Generator/third_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpM^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
agradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
8gradients_1/Generator/second_layer/leaky_relu_grad/ShapeShape%Generator/second_layer/leaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2ShapeVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
8gradients_1/Generator/second_layer/leaky_relu_grad/zerosFill:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Hgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/Generator/second_layer/leaky_relu_grad/Shape:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
9gradients_1/Generator/second_layer/leaky_relu_grad/SelectSelect?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency8gradients_1/Generator/second_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Select?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqual8gradients_1/Generator/second_layer/leaky_relu_grad/zerosVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
6gradients_1/Generator/second_layer/leaky_relu_grad/SumSum9gradients_1/Generator/second_layer/leaky_relu_grad/SelectHgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeReshape6gradients_1/Generator/second_layer/leaky_relu_grad/Sum8gradients_1/Generator/second_layer/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1Sum;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Jgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1Reshape8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Cgradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_depsNoOp;^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape=^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1
�
Kgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeD^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*M
_classC
A?loc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape
�
Mgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Identity<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1D^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������

<gradients_1/Generator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Lgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulMulKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency:Generator/second_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/SumSum:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulLgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeReshape:gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Mul'Generator/second_layer/leaky_relu/alphaKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1Sum<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Mul_1Ngradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Ggradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp?^gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeA^gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
�
Ogradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeH^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*Q
_classG
ECloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape
�
Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1H^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*S
_classI
GEloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
�
gradients_1/AddN_6AddNMgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape:Generator/second_layer/batch_normalization/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_6agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6cgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape
�
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape.Generator/second_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
agradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency8Generator/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumSumOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mulagradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul.Generator/second_layer/fully_connected/BiasAdddgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1cgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/NegNegfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpg^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1N^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
�
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Kgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGraddgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Pgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpe^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyL^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Xgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitydgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
Zgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_18Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Muldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1;Generator/second_layer/batch_normalization/moving_mean/read*
T0*
_output_shapes	
:�
�
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_depsNoOpP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/MulR^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
�
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Egradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulMatMulXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency2Generator/second_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Ggradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/first_layer/leaky_reluXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Ogradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpF^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulH^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1
�
Wgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityEgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulP^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*X
_classN
LJloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul
�
Ygradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityGgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1P^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*Z
_classP
NLloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1
�
gradients_1/AddN_7AddNfgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_75Generator/second_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_7:Generator/second_layer/batch_normalization/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpN^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulP^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
7gradients_1/Generator/first_layer/leaky_relu_grad/ShapeShape$Generator/first_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2ShapeWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
7gradients_1/Generator/first_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Ggradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/first_layer/leaky_relu_grad/Shape9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8gradients_1/Generator/first_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/first_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/first_layer/leaky_relu_grad/zerosWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
5gradients_1/Generator/first_layer/leaky_relu_grad/SumSum8gradients_1/Generator/first_layer/leaky_relu_grad/SelectGgradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/first_layer/leaky_relu_grad/Sum7gradients_1/Generator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Igradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_19gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Bgradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
�
Jgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*L
_classB
@>loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape
�
Lgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1C^gradients_1/Generator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
~
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Kgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulMulJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/first_layer/leaky_relu/alphaJgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Fgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
�
Ngradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_8AddNLgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Jgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Ogradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_8K^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Wgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8P^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Ygradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Dgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/first_layer/fully_connected/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(
�
Fgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise_inWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d�*
transpose_a(*
transpose_b( 
�
Ngradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
Vgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
Xgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/group_deps*
_output_shapes
:	d�*
T0*Y
_classO
MKloc:@gradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1
�
beta1_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power_1
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container 
�
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
�
beta2_power_1/initial_valueConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
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
�
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
�
SGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   �   
�
IGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    
�
CGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	d�*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0
�
1Generator/first_layer/fully_connected/kernel/Adam
VariableV2*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
�
8Generator/first_layer/fully_connected/kernel/Adam/AssignAssign1Generator/first_layer/fully_connected/kernel/AdamCGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
�
6Generator/first_layer/fully_connected/kernel/Adam/readIdentity1Generator/first_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
UGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
�
KGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
3Generator/first_layer/fully_connected/kernel/Adam_1
VariableV2*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
�
:Generator/first_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/first_layer/fully_connected/kernel/Adam_1EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(
�
8Generator/first_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/first_layer/fully_connected/kernel/Adam_1*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
AGenerator/first_layer/fully_connected/bias/Adam/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
/Generator/first_layer/fully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:�
�
6Generator/first_layer/fully_connected/bias/Adam/AssignAssign/Generator/first_layer/fully_connected/bias/AdamAGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
4Generator/first_layer/fully_connected/bias/Adam/readIdentity/Generator/first_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:�
�
CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
1Generator/first_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:�
�
8Generator/first_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/first_layer/fully_connected/bias/Adam_1CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
6Generator/first_layer/fully_connected/bias/Adam_1/readIdentity1Generator/first_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:�
�
TGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
JGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
DGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillTGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorJGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0
�
2Generator/second_layer/fully_connected/kernel/Adam
VariableV2*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
9Generator/second_layer/fully_connected/kernel/Adam/AssignAssign2Generator/second_layer/fully_connected/kernel/AdamDGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
�
7Generator/second_layer/fully_connected/kernel/Adam/readIdentity2Generator/second_layer/fully_connected/kernel/Adam* 
_output_shapes
:
��*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
�
VGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
LGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillVGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
4Generator/second_layer/fully_connected/kernel/Adam_1
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container 
�
;Generator/second_layer/fully_connected/kernel/Adam_1/AssignAssign4Generator/second_layer/fully_connected/kernel/Adam_1FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
9Generator/second_layer/fully_connected/kernel/Adam_1/readIdentity4Generator/second_layer/fully_connected/kernel/Adam_1*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��
�
BGenerator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB�*    
�
0Generator/second_layer/fully_connected/bias/Adam
VariableV2*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
7Generator/second_layer/fully_connected/bias/Adam/AssignAssign0Generator/second_layer/fully_connected/bias/AdamBGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias
�
5Generator/second_layer/fully_connected/bias/Adam/readIdentity0Generator/second_layer/fully_connected/bias/Adam*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
2Generator/second_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:�
�
9Generator/second_layer/fully_connected/bias/Adam_1/AssignAssign2Generator/second_layer/fully_connected/bias/Adam_1DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
7Generator/second_layer/fully_connected/bias/Adam_1/readIdentity2Generator/second_layer/fully_connected/bias/Adam_1*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
GGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB�*    
�
5Generator/second_layer/batch_normalization/gamma/Adam
VariableV2*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
<Generator/second_layer/batch_normalization/gamma/Adam/AssignAssign5Generator/second_layer/batch_normalization/gamma/AdamGGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
:Generator/second_layer/batch_normalization/gamma/Adam/readIdentity5Generator/second_layer/batch_normalization/gamma/Adam*
_output_shapes	
:�*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
�
IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7Generator/second_layer/batch_normalization/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:�
�
>Generator/second_layer/batch_normalization/gamma/Adam_1/AssignAssign7Generator/second_layer/batch_normalization/gamma/Adam_1IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
<Generator/second_layer/batch_normalization/gamma/Adam_1/readIdentity7Generator/second_layer/batch_normalization/gamma/Adam_1*
_output_shapes	
:�*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
�
FGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4Generator/second_layer/batch_normalization/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:�
�
;Generator/second_layer/batch_normalization/beta/Adam/AssignAssign4Generator/second_layer/batch_normalization/beta/AdamFGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
�
9Generator/second_layer/batch_normalization/beta/Adam/readIdentity4Generator/second_layer/batch_normalization/beta/Adam*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
6Generator/second_layer/batch_normalization/beta/Adam_1
VariableV2*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
=Generator/second_layer/batch_normalization/beta/Adam_1/AssignAssign6Generator/second_layer/batch_normalization/beta/Adam_1HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
;Generator/second_layer/batch_normalization/beta/Adam_1/readIdentity6Generator/second_layer/batch_normalization/beta/Adam_1*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
_output_shapes	
:�
�
SGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
IGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
CGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zerosFillSGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorIGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
1Generator/third_layer/fully_connected/kernel/Adam
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container 
�
8Generator/third_layer/fully_connected/kernel/Adam/AssignAssign1Generator/third_layer/fully_connected/kernel/AdamCGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
6Generator/third_layer/fully_connected/kernel/Adam/readIdentity1Generator/third_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��
�
UGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
KGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    
�
EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
3Generator/third_layer/fully_connected/kernel/Adam_1
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container 
�
:Generator/third_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/third_layer/fully_connected/kernel/Adam_1EGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
8Generator/third_layer/fully_connected/kernel/Adam_1/readIdentity3Generator/third_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
�
AGenerator/third_layer/fully_connected/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB�*    
�
/Generator/third_layer/fully_connected/bias/Adam
VariableV2*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
6Generator/third_layer/fully_connected/bias/Adam/AssignAssign/Generator/third_layer/fully_connected/bias/AdamAGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
4Generator/third_layer/fully_connected/bias/Adam/readIdentity/Generator/third_layer/fully_connected/bias/Adam*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:�
�
CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
valueB�*    
�
1Generator/third_layer/fully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:�
�
8Generator/third_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/third_layer/fully_connected/bias/Adam_1CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
6Generator/third_layer/fully_connected/bias/Adam_1/readIdentity1Generator/third_layer/fully_connected/bias/Adam_1*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:�
�
FGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4Generator/third_layer/batch_normalization/gamma/Adam
VariableV2*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
;Generator/third_layer/batch_normalization/gamma/Adam/AssignAssign4Generator/third_layer/batch_normalization/gamma/AdamFGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
9Generator/third_layer/batch_normalization/gamma/Adam/readIdentity4Generator/third_layer/batch_normalization/gamma/Adam*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:�
�
HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zerosConst*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
6Generator/third_layer/batch_normalization/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
	container *
shape:�
�
=Generator/third_layer/batch_normalization/gamma/Adam_1/AssignAssign6Generator/third_layer/batch_normalization/gamma/Adam_1HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
;Generator/third_layer/batch_normalization/gamma/Adam_1/readIdentity6Generator/third_layer/batch_normalization/gamma/Adam_1*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:�
�
EGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3Generator/third_layer/batch_normalization/beta/Adam
VariableV2*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
:Generator/third_layer/batch_normalization/beta/Adam/AssignAssign3Generator/third_layer/batch_normalization/beta/AdamEGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
8Generator/third_layer/batch_normalization/beta/Adam/readIdentity3Generator/third_layer/batch_normalization/beta/Adam*
_output_shapes	
:�*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
�
GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5Generator/third_layer/batch_normalization/beta/Adam_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
�
<Generator/third_layer/batch_normalization/beta/Adam_1/AssignAssign5Generator/third_layer/batch_normalization/beta/Adam_1GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
:Generator/third_layer/batch_normalization/beta/Adam_1/readIdentity5Generator/third_layer/batch_normalization/beta/Adam_1*
_output_shapes	
:�*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta
�
RGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
HGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    
�
BGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zerosFillRGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
0Generator/last_layer/fully_connected/kernel/Adam
VariableV2*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
7Generator/last_layer/fully_connected/kernel/Adam/AssignAssign0Generator/last_layer/fully_connected/kernel/AdamBGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel
�
5Generator/last_layer/fully_connected/kernel/Adam/readIdentity0Generator/last_layer/fully_connected/kernel/Adam*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��
�
TGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      
�
JGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillTGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorJGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0
�
2Generator/last_layer/fully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
��
�
9Generator/last_layer/fully_connected/kernel/Adam_1/AssignAssign2Generator/last_layer/fully_connected/kernel/Adam_1DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
7Generator/last_layer/fully_connected/kernel/Adam_1/readIdentity2Generator/last_layer/fully_connected/kernel/Adam_1*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��
�
PGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
FGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@Generator/last_layer/fully_connected/bias/Adam/Initializer/zerosFillPGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorFGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:�
�
.Generator/last_layer/fully_connected/bias/Adam
VariableV2*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
5Generator/last_layer/fully_connected/bias/Adam/AssignAssign.Generator/last_layer/fully_connected/bias/Adam@Generator/last_layer/fully_connected/bias/Adam/Initializer/zeros*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
3Generator/last_layer/fully_connected/bias/Adam/readIdentity.Generator/last_layer/fully_connected/bias/Adam*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:�
�
RGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
HGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/ConstConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zerosFillRGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/Const*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:�
�
0Generator/last_layer/fully_connected/bias/Adam_1
VariableV2*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
7Generator/last_layer/fully_connected/bias/Adam_1/AssignAssign0Generator/last_layer/fully_connected/bias/Adam_1BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
�
5Generator/last_layer/fully_connected/bias/Adam_1/readIdentity0Generator/last_layer/fully_connected/bias/Adam_1*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:�
�
UGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:�*
dtype0*
_output_shapes
:
�
KGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
�
EGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zerosFillUGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorKGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:�
�
3Generator/last_layer/batch_normalization/gamma/Adam
VariableV2*
shared_name *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
:Generator/last_layer/batch_normalization/gamma/Adam/AssignAssign3Generator/last_layer/batch_normalization/gamma/AdamEGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
8Generator/last_layer/batch_normalization/gamma/Adam/readIdentity3Generator/last_layer/batch_normalization/gamma/Adam*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:�
�
WGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:�
�
MGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/ConstConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
�
GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zerosFillWGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorMGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/Const*
_output_shapes	
:�*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0
�
5Generator/last_layer/batch_normalization/gamma/Adam_1
VariableV2*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
<Generator/last_layer/batch_normalization/gamma/Adam_1/AssignAssign5Generator/last_layer/batch_normalization/gamma/Adam_1GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
:Generator/last_layer/batch_normalization/gamma/Adam_1/readIdentity5Generator/last_layer/batch_normalization/gamma/Adam_1*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:�
�
TGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:�*
dtype0*
_output_shapes
:
�
JGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
DGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zerosFillTGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorJGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/Const*
_output_shapes	
:�*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0
�
2Generator/last_layer/batch_normalization/beta/Adam
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
�
9Generator/last_layer/batch_normalization/beta/Adam/AssignAssign2Generator/last_layer/batch_normalization/beta/AdamDGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
�
7Generator/last_layer/batch_normalization/beta/Adam/readIdentity2Generator/last_layer/batch_normalization/beta/Adam*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:�
�
VGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:�*
dtype0*
_output_shapes
:
�
LGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/ConstConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zerosFillVGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:�
�
4Generator/last_layer/batch_normalization/beta/Adam_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
�
;Generator/last_layer/batch_normalization/beta/Adam_1/AssignAssign4Generator/last_layer/batch_normalization/beta/Adam_1FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
9Generator/last_layer/batch_normalization/beta/Adam_1/readIdentity4Generator/last_layer/batch_normalization/beta/Adam_1*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:�
�
BGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
8Generator/fake_image/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    
�
2Generator/fake_image/kernel/Adam/Initializer/zerosFillBGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensor8Generator/fake_image/kernel/Adam/Initializer/zeros/Const*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:
��
�
 Generator/fake_image/kernel/Adam
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel
�
'Generator/fake_image/kernel/Adam/AssignAssign Generator/fake_image/kernel/Adam2Generator/fake_image/kernel/Adam/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
��
�
%Generator/fake_image/kernel/Adam/readIdentity Generator/fake_image/kernel/Adam* 
_output_shapes
:
��*
T0*.
_class$
" loc:@Generator/fake_image/kernel
�
DGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
:Generator/fake_image/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    
�
4Generator/fake_image/kernel/Adam_1/Initializer/zerosFillDGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensor:Generator/fake_image/kernel/Adam_1/Initializer/zeros/Const*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0* 
_output_shapes
:
��
�
"Generator/fake_image/kernel/Adam_1
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel
�
)Generator/fake_image/kernel/Adam_1/AssignAssign"Generator/fake_image/kernel/Adam_14Generator/fake_image/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel
�
'Generator/fake_image/kernel/Adam_1/readIdentity"Generator/fake_image/kernel/Adam_1*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��
�
0Generator/fake_image/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*,
_class"
 loc:@Generator/fake_image/bias*
valueB�*    
�
Generator/fake_image/bias/Adam
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *,
_class"
 loc:@Generator/fake_image/bias
�
%Generator/fake_image/bias/Adam/AssignAssignGenerator/fake_image/bias/Adam0Generator/fake_image/bias/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:�
�
#Generator/fake_image/bias/Adam/readIdentityGenerator/fake_image/bias/Adam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:�
�
2Generator/fake_image/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
 Generator/fake_image/bias/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
'Generator/fake_image/bias/Adam_1/AssignAssign Generator/fake_image/bias/Adam_12Generator/fake_image/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:�
�
%Generator/fake_image/bias/Adam_1/readIdentity Generator/fake_image/bias/Adam_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:�
Y
Adam_1/learning_rateConst*
valueB
 *�Q9*
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
 *w�?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
DAdam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/first_layer/fully_connected/kernel1Generator/first_layer/fully_connected/kernel/Adam3Generator/first_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	d�*
use_locking( *
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
�
BAdam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/first_layer/fully_connected/bias/Generator/first_layer/fully_connected/bias/Adam1Generator/first_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
EAdam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam-Generator/second_layer/fully_connected/kernel2Generator/second_layer/fully_connected/kernel/Adam4Generator/second_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
CAdam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam+Generator/second_layer/fully_connected/bias0Generator/second_layer/fully_connected/bias/Adam2Generator/second_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonZgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
HAdam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam0Generator/second_layer/batch_normalization/gamma5Generator/second_layer/batch_normalization/gamma/Adam7Generator/second_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�
�
GAdam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam/Generator/second_layer/batch_normalization/beta4Generator/second_layer/batch_normalization/beta/Adam6Generator/second_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
�
DAdam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam	ApplyAdam,Generator/third_layer/fully_connected/kernel1Generator/third_layer/fully_connected/kernel/Adam3Generator/third_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
�
BAdam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/third_layer/fully_connected/bias/Generator/third_layer/fully_connected/bias/Adam1Generator/third_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
GAdam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam/Generator/third_layer/batch_normalization/gamma4Generator/third_layer/batch_normalization/gamma/Adam6Generator/third_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
FAdam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdam	ApplyAdam.Generator/third_layer/batch_normalization/beta3Generator/third_layer/batch_normalization/beta/Adam5Generator/third_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
CAdam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Generator/last_layer/fully_connected/kernel0Generator/last_layer/fully_connected/kernel/Adam2Generator/last_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
AAdam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Generator/last_layer/fully_connected/bias.Generator/last_layer/fully_connected/bias/Adam0Generator/last_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
FAdam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Generator/last_layer/batch_normalization/gamma3Generator/last_layer/batch_normalization/gamma/Adam5Generator/last_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�
�
EAdam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Generator/last_layer/batch_normalization/beta2Generator/last_layer/batch_normalization/beta/Adam4Generator/last_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:�
�
3Adam_1/update_Generator/fake_image/kernel/ApplyAdam	ApplyAdamGenerator/fake_image/kernel Generator/fake_image/kernel/Adam"Generator/fake_image/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
�
1Adam_1/update_Generator/fake_image/bias/ApplyAdam	ApplyAdamGenerator/fake_image/biasGenerator/fake_image/bias/Adam Generator/fake_image/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�	
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta22^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*,
_class"
 loc:@Generator/fake_image/bias
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
�	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: ""��
	variables����
�
.Generator/first_layer/fully_connected/kernel:03Generator/first_layer/fully_connected/kernel/Assign3Generator/first_layer/fully_connected/kernel/read:02IGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
�
,Generator/first_layer/fully_connected/bias:01Generator/first_layer/fully_connected/bias/Assign1Generator/first_layer/fully_connected/bias/read:02>Generator/first_layer/fully_connected/bias/Initializer/zeros:08
�
/Generator/second_layer/fully_connected/kernel:04Generator/second_layer/fully_connected/kernel/Assign4Generator/second_layer/fully_connected/kernel/read:02JGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
�
-Generator/second_layer/fully_connected/bias:02Generator/second_layer/fully_connected/bias/Assign2Generator/second_layer/fully_connected/bias/read:02?Generator/second_layer/fully_connected/bias/Initializer/zeros:08
�
2Generator/second_layer/batch_normalization/gamma:07Generator/second_layer/batch_normalization/gamma/Assign7Generator/second_layer/batch_normalization/gamma/read:02CGenerator/second_layer/batch_normalization/gamma/Initializer/ones:08
�
1Generator/second_layer/batch_normalization/beta:06Generator/second_layer/batch_normalization/beta/Assign6Generator/second_layer/batch_normalization/beta/read:02CGenerator/second_layer/batch_normalization/beta/Initializer/zeros:08
�
8Generator/second_layer/batch_normalization/moving_mean:0=Generator/second_layer/batch_normalization/moving_mean/Assign=Generator/second_layer/batch_normalization/moving_mean/read:02JGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros:0
�
<Generator/second_layer/batch_normalization/moving_variance:0AGenerator/second_layer/batch_normalization/moving_variance/AssignAGenerator/second_layer/batch_normalization/moving_variance/read:02MGenerator/second_layer/batch_normalization/moving_variance/Initializer/ones:0
�
.Generator/third_layer/fully_connected/kernel:03Generator/third_layer/fully_connected/kernel/Assign3Generator/third_layer/fully_connected/kernel/read:02IGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform:08
�
,Generator/third_layer/fully_connected/bias:01Generator/third_layer/fully_connected/bias/Assign1Generator/third_layer/fully_connected/bias/read:02>Generator/third_layer/fully_connected/bias/Initializer/zeros:08
�
1Generator/third_layer/batch_normalization/gamma:06Generator/third_layer/batch_normalization/gamma/Assign6Generator/third_layer/batch_normalization/gamma/read:02BGenerator/third_layer/batch_normalization/gamma/Initializer/ones:08
�
0Generator/third_layer/batch_normalization/beta:05Generator/third_layer/batch_normalization/beta/Assign5Generator/third_layer/batch_normalization/beta/read:02BGenerator/third_layer/batch_normalization/beta/Initializer/zeros:08
�
7Generator/third_layer/batch_normalization/moving_mean:0<Generator/third_layer/batch_normalization/moving_mean/Assign<Generator/third_layer/batch_normalization/moving_mean/read:02IGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros:0
�
;Generator/third_layer/batch_normalization/moving_variance:0@Generator/third_layer/batch_normalization/moving_variance/Assign@Generator/third_layer/batch_normalization/moving_variance/read:02LGenerator/third_layer/batch_normalization/moving_variance/Initializer/ones:0
�
-Generator/last_layer/fully_connected/kernel:02Generator/last_layer/fully_connected/kernel/Assign2Generator/last_layer/fully_connected/kernel/read:02HGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform:08
�
+Generator/last_layer/fully_connected/bias:00Generator/last_layer/fully_connected/bias/Assign0Generator/last_layer/fully_connected/bias/read:02=Generator/last_layer/fully_connected/bias/Initializer/zeros:08
�
0Generator/last_layer/batch_normalization/gamma:05Generator/last_layer/batch_normalization/gamma/Assign5Generator/last_layer/batch_normalization/gamma/read:02AGenerator/last_layer/batch_normalization/gamma/Initializer/ones:08
�
/Generator/last_layer/batch_normalization/beta:04Generator/last_layer/batch_normalization/beta/Assign4Generator/last_layer/batch_normalization/beta/read:02AGenerator/last_layer/batch_normalization/beta/Initializer/zeros:08
�
6Generator/last_layer/batch_normalization/moving_mean:0;Generator/last_layer/batch_normalization/moving_mean/Assign;Generator/last_layer/batch_normalization/moving_mean/read:02HGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros:0
�
:Generator/last_layer/batch_normalization/moving_variance:0?Generator/last_layer/batch_normalization/moving_variance/Assign?Generator/last_layer/batch_normalization/moving_variance/read:02KGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones:0
�
Generator/fake_image/kernel:0"Generator/fake_image/kernel/Assign"Generator/fake_image/kernel/read:028Generator/fake_image/kernel/Initializer/random_uniform:08
�
Generator/fake_image/bias:0 Generator/fake_image/bias/Assign Generator/fake_image/bias/read:02-Generator/fake_image/bias/Initializer/zeros:08
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
7Discriminator/first_layer/fully_connected/kernel/Adam:0<Discriminator/first_layer/fully_connected/kernel/Adam/Assign<Discriminator/first_layer/fully_connected/kernel/Adam/read:02IDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros:0
�
9Discriminator/first_layer/fully_connected/kernel/Adam_1:0>Discriminator/first_layer/fully_connected/kernel/Adam_1/Assign>Discriminator/first_layer/fully_connected/kernel/Adam_1/read:02KDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
�
5Discriminator/first_layer/fully_connected/bias/Adam:0:Discriminator/first_layer/fully_connected/bias/Adam/Assign:Discriminator/first_layer/fully_connected/bias/Adam/read:02GDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros:0
�
7Discriminator/first_layer/fully_connected/bias/Adam_1:0<Discriminator/first_layer/fully_connected/bias/Adam_1/Assign<Discriminator/first_layer/fully_connected/bias/Adam_1/read:02IDiscriminator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
�
8Discriminator/second_layer/fully_connected/kernel/Adam:0=Discriminator/second_layer/fully_connected/kernel/Adam/Assign=Discriminator/second_layer/fully_connected/kernel/Adam/read:02JDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros:0
�
:Discriminator/second_layer/fully_connected/kernel/Adam_1:0?Discriminator/second_layer/fully_connected/kernel/Adam_1/Assign?Discriminator/second_layer/fully_connected/kernel/Adam_1/read:02LDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
�
6Discriminator/second_layer/fully_connected/bias/Adam:0;Discriminator/second_layer/fully_connected/bias/Adam/Assign;Discriminator/second_layer/fully_connected/bias/Adam/read:02HDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros:0
�
8Discriminator/second_layer/fully_connected/bias/Adam_1:0=Discriminator/second_layer/fully_connected/bias/Adam_1/Assign=Discriminator/second_layer/fully_connected/bias/Adam_1/read:02JDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
�
 Discriminator/prob/kernel/Adam:0%Discriminator/prob/kernel/Adam/Assign%Discriminator/prob/kernel/Adam/read:022Discriminator/prob/kernel/Adam/Initializer/zeros:0
�
"Discriminator/prob/kernel/Adam_1:0'Discriminator/prob/kernel/Adam_1/Assign'Discriminator/prob/kernel/Adam_1/read:024Discriminator/prob/kernel/Adam_1/Initializer/zeros:0
�
Discriminator/prob/bias/Adam:0#Discriminator/prob/bias/Adam/Assign#Discriminator/prob/bias/Adam/read:020Discriminator/prob/bias/Adam/Initializer/zeros:0
�
 Discriminator/prob/bias/Adam_1:0%Discriminator/prob/bias/Adam_1/Assign%Discriminator/prob/bias/Adam_1/read:022Discriminator/prob/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
�
3Generator/first_layer/fully_connected/kernel/Adam:08Generator/first_layer/fully_connected/kernel/Adam/Assign8Generator/first_layer/fully_connected/kernel/Adam/read:02EGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros:0
�
5Generator/first_layer/fully_connected/kernel/Adam_1:0:Generator/first_layer/fully_connected/kernel/Adam_1/Assign:Generator/first_layer/fully_connected/kernel/Adam_1/read:02GGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
�
1Generator/first_layer/fully_connected/bias/Adam:06Generator/first_layer/fully_connected/bias/Adam/Assign6Generator/first_layer/fully_connected/bias/Adam/read:02CGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros:0
�
3Generator/first_layer/fully_connected/bias/Adam_1:08Generator/first_layer/fully_connected/bias/Adam_1/Assign8Generator/first_layer/fully_connected/bias/Adam_1/read:02EGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
�
4Generator/second_layer/fully_connected/kernel/Adam:09Generator/second_layer/fully_connected/kernel/Adam/Assign9Generator/second_layer/fully_connected/kernel/Adam/read:02FGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros:0
�
6Generator/second_layer/fully_connected/kernel/Adam_1:0;Generator/second_layer/fully_connected/kernel/Adam_1/Assign;Generator/second_layer/fully_connected/kernel/Adam_1/read:02HGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
�
2Generator/second_layer/fully_connected/bias/Adam:07Generator/second_layer/fully_connected/bias/Adam/Assign7Generator/second_layer/fully_connected/bias/Adam/read:02DGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros:0
�
4Generator/second_layer/fully_connected/bias/Adam_1:09Generator/second_layer/fully_connected/bias/Adam_1/Assign9Generator/second_layer/fully_connected/bias/Adam_1/read:02FGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
�
7Generator/second_layer/batch_normalization/gamma/Adam:0<Generator/second_layer/batch_normalization/gamma/Adam/Assign<Generator/second_layer/batch_normalization/gamma/Adam/read:02IGenerator/second_layer/batch_normalization/gamma/Adam/Initializer/zeros:0
�
9Generator/second_layer/batch_normalization/gamma/Adam_1:0>Generator/second_layer/batch_normalization/gamma/Adam_1/Assign>Generator/second_layer/batch_normalization/gamma/Adam_1/read:02KGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
�
6Generator/second_layer/batch_normalization/beta/Adam:0;Generator/second_layer/batch_normalization/beta/Adam/Assign;Generator/second_layer/batch_normalization/beta/Adam/read:02HGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros:0
�
8Generator/second_layer/batch_normalization/beta/Adam_1:0=Generator/second_layer/batch_normalization/beta/Adam_1/Assign=Generator/second_layer/batch_normalization/beta/Adam_1/read:02JGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
�
3Generator/third_layer/fully_connected/kernel/Adam:08Generator/third_layer/fully_connected/kernel/Adam/Assign8Generator/third_layer/fully_connected/kernel/Adam/read:02EGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros:0
�
5Generator/third_layer/fully_connected/kernel/Adam_1:0:Generator/third_layer/fully_connected/kernel/Adam_1/Assign:Generator/third_layer/fully_connected/kernel/Adam_1/read:02GGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
�
1Generator/third_layer/fully_connected/bias/Adam:06Generator/third_layer/fully_connected/bias/Adam/Assign6Generator/third_layer/fully_connected/bias/Adam/read:02CGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros:0
�
3Generator/third_layer/fully_connected/bias/Adam_1:08Generator/third_layer/fully_connected/bias/Adam_1/Assign8Generator/third_layer/fully_connected/bias/Adam_1/read:02EGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
�
6Generator/third_layer/batch_normalization/gamma/Adam:0;Generator/third_layer/batch_normalization/gamma/Adam/Assign;Generator/third_layer/batch_normalization/gamma/Adam/read:02HGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros:0
�
8Generator/third_layer/batch_normalization/gamma/Adam_1:0=Generator/third_layer/batch_normalization/gamma/Adam_1/Assign=Generator/third_layer/batch_normalization/gamma/Adam_1/read:02JGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
�
5Generator/third_layer/batch_normalization/beta/Adam:0:Generator/third_layer/batch_normalization/beta/Adam/Assign:Generator/third_layer/batch_normalization/beta/Adam/read:02GGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros:0
�
7Generator/third_layer/batch_normalization/beta/Adam_1:0<Generator/third_layer/batch_normalization/beta/Adam_1/Assign<Generator/third_layer/batch_normalization/beta/Adam_1/read:02IGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
�
2Generator/last_layer/fully_connected/kernel/Adam:07Generator/last_layer/fully_connected/kernel/Adam/Assign7Generator/last_layer/fully_connected/kernel/Adam/read:02DGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros:0
�
4Generator/last_layer/fully_connected/kernel/Adam_1:09Generator/last_layer/fully_connected/kernel/Adam_1/Assign9Generator/last_layer/fully_connected/kernel/Adam_1/read:02FGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros:0
�
0Generator/last_layer/fully_connected/bias/Adam:05Generator/last_layer/fully_connected/bias/Adam/Assign5Generator/last_layer/fully_connected/bias/Adam/read:02BGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros:0
�
2Generator/last_layer/fully_connected/bias/Adam_1:07Generator/last_layer/fully_connected/bias/Adam_1/Assign7Generator/last_layer/fully_connected/bias/Adam_1/read:02DGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros:0
�
5Generator/last_layer/batch_normalization/gamma/Adam:0:Generator/last_layer/batch_normalization/gamma/Adam/Assign:Generator/last_layer/batch_normalization/gamma/Adam/read:02GGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros:0
�
7Generator/last_layer/batch_normalization/gamma/Adam_1:0<Generator/last_layer/batch_normalization/gamma/Adam_1/Assign<Generator/last_layer/batch_normalization/gamma/Adam_1/read:02IGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros:0
�
4Generator/last_layer/batch_normalization/beta/Adam:09Generator/last_layer/batch_normalization/beta/Adam/Assign9Generator/last_layer/batch_normalization/beta/Adam/read:02FGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros:0
�
6Generator/last_layer/batch_normalization/beta/Adam_1:0;Generator/last_layer/batch_normalization/beta/Adam_1/Assign;Generator/last_layer/batch_normalization/beta/Adam_1/read:02HGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros:0
�
"Generator/fake_image/kernel/Adam:0'Generator/fake_image/kernel/Adam/Assign'Generator/fake_image/kernel/Adam/read:024Generator/fake_image/kernel/Adam/Initializer/zeros:0
�
$Generator/fake_image/kernel/Adam_1:0)Generator/fake_image/kernel/Adam_1/Assign)Generator/fake_image/kernel/Adam_1/read:026Generator/fake_image/kernel/Adam_1/Initializer/zeros:0
�
 Generator/fake_image/bias/Adam:0%Generator/fake_image/bias/Adam/Assign%Generator/fake_image/bias/Adam/read:022Generator/fake_image/bias/Adam/Initializer/zeros:0
�
"Generator/fake_image/bias/Adam_1:0'Generator/fake_image/bias/Adam_1/Assign'Generator/fake_image/bias/Adam_1/read:024Generator/fake_image/bias/Adam_1/Initializer/zeros:0"7
	summaries*
(
discriminator_loss:0
generator_loss:0"�%
trainable_variables�%�%
�
.Generator/first_layer/fully_connected/kernel:03Generator/first_layer/fully_connected/kernel/Assign3Generator/first_layer/fully_connected/kernel/read:02IGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform:08
�
,Generator/first_layer/fully_connected/bias:01Generator/first_layer/fully_connected/bias/Assign1Generator/first_layer/fully_connected/bias/read:02>Generator/first_layer/fully_connected/bias/Initializer/zeros:08
�
/Generator/second_layer/fully_connected/kernel:04Generator/second_layer/fully_connected/kernel/Assign4Generator/second_layer/fully_connected/kernel/read:02JGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform:08
�
-Generator/second_layer/fully_connected/bias:02Generator/second_layer/fully_connected/bias/Assign2Generator/second_layer/fully_connected/bias/read:02?Generator/second_layer/fully_connected/bias/Initializer/zeros:08
�
2Generator/second_layer/batch_normalization/gamma:07Generator/second_layer/batch_normalization/gamma/Assign7Generator/second_layer/batch_normalization/gamma/read:02CGenerator/second_layer/batch_normalization/gamma/Initializer/ones:08
�
1Generator/second_layer/batch_normalization/beta:06Generator/second_layer/batch_normalization/beta/Assign6Generator/second_layer/batch_normalization/beta/read:02CGenerator/second_layer/batch_normalization/beta/Initializer/zeros:08
�
.Generator/third_layer/fully_connected/kernel:03Generator/third_layer/fully_connected/kernel/Assign3Generator/third_layer/fully_connected/kernel/read:02IGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform:08
�
,Generator/third_layer/fully_connected/bias:01Generator/third_layer/fully_connected/bias/Assign1Generator/third_layer/fully_connected/bias/read:02>Generator/third_layer/fully_connected/bias/Initializer/zeros:08
�
1Generator/third_layer/batch_normalization/gamma:06Generator/third_layer/batch_normalization/gamma/Assign6Generator/third_layer/batch_normalization/gamma/read:02BGenerator/third_layer/batch_normalization/gamma/Initializer/ones:08
�
0Generator/third_layer/batch_normalization/beta:05Generator/third_layer/batch_normalization/beta/Assign5Generator/third_layer/batch_normalization/beta/read:02BGenerator/third_layer/batch_normalization/beta/Initializer/zeros:08
�
-Generator/last_layer/fully_connected/kernel:02Generator/last_layer/fully_connected/kernel/Assign2Generator/last_layer/fully_connected/kernel/read:02HGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform:08
�
+Generator/last_layer/fully_connected/bias:00Generator/last_layer/fully_connected/bias/Assign0Generator/last_layer/fully_connected/bias/read:02=Generator/last_layer/fully_connected/bias/Initializer/zeros:08
�
0Generator/last_layer/batch_normalization/gamma:05Generator/last_layer/batch_normalization/gamma/Assign5Generator/last_layer/batch_normalization/gamma/read:02AGenerator/last_layer/batch_normalization/gamma/Initializer/ones:08
�
/Generator/last_layer/batch_normalization/beta:04Generator/last_layer/batch_normalization/beta/Assign4Generator/last_layer/batch_normalization/beta/read:02AGenerator/last_layer/batch_normalization/beta/Initializer/zeros:08
�
Generator/fake_image/kernel:0"Generator/fake_image/kernel/Assign"Generator/fake_image/kernel/read:028Generator/fake_image/kernel/Initializer/random_uniform:08
�
Generator/fake_image/bias:0 Generator/fake_image/bias/Assign Generator/fake_image/bias/read:02-Generator/fake_image/bias/Initializer/zeros:08
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
Discriminator/prob/bias:0Discriminator/prob/bias/AssignDiscriminator/prob/bias/read:02+Discriminator/prob/bias/Initializer/zeros:08"
train_op

Adam
Adam_1Ð��       �N�	j�d����A*�
w
discriminator_loss*a	   �B��?   �B��?      �?!   �B��?)�<B?�@2�E̟���?yL�����?�������:              �?        
s
generator_loss*a	   ��O�   ��O�      �?!   ��O�) ����N�?2+Se*8�\l�9��������:              �?        ��qB�       �{�	z������A(*�
w
discriminator_loss*a	   �04�?   �04�?      �?!   �04�?)�D�n��h?2���g��?I���?�������:              �?        
s
generator_loss*a	   @��p�   @��p�      �?!   @��p�) A���>2uWy��r�;8�clp��������:              �?        �87�       �{�	�������AP*�
w
discriminator_loss*a	   �ݑ�?   �ݑ�?      �?!   �ݑ�?)@��DS�?2-Ա�L�?eiS�m�?�������:              �?        
s
generator_loss*a	   ���>�   ���>�      �?!   ���>�) i;��F�>2���#@�d�\D�X=��������:              �?        ��5)�       �{�	5�"����Ax*�
w
discriminator_loss*a	   @4g?   @4g?      �?!   @4g?)�P��{��>2Tw��Nof?P}���h?�������:              �?        
s
generator_loss*a	    ��/�    ��/�      �?!    ��/�)  ��ro>2��bȬ�0���VlQ.��������:              �?        ����       b�D�	ye����A�*�
w
discriminator_loss*a	    ��k?    ��k?      �?!    ��k?) �k�H��>2ߤ�(g%k?�N�W�m?�������:              �?        
s
generator_loss*a	   @�AC�   @�AC�      �?!   @�AC�) I���,�>2�T���C��!�A��������:              �?        ����       b�D�	�
�����A�*�
w
discriminator_loss*a	   �-k?   �-k?      �?!   �-k?) �c�>2ߤ�(g%k?�N�W�m?�������:              �?        
s
generator_loss*a	   ��3_�   ��3_�      �?!   ��3_�)���rfl�>2�l�P�`�E��{��^��������:              �?        F�A��       b�D�	y������A�*�
w
discriminator_loss*a	    ���?    ���?      �?!    ���?) ������?22g�G�A�?������?�������:              �?        
s
generator_loss*a	   �X��   �X��      �?!   �X��) �LD�N�?2������2g�G�A��������:              �?        ���       �E��	�;,����A�*�
w
discriminator_loss*a	   `�^`?   `�^`?      �?!   `�^`?)@��䆿�>2E��{��^?�l�P�`?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��U�       �E��	�\m����A�*�
w
discriminator_loss*a	   ���K?   ���K?      �?!   ���K?) 2'���>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �b>�       �E��	 �����A�*�
w
discriminator_loss*a	   ���T?   ���T?      �?!   ���T?)@胅���>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        06�`�       �E��	�S�����A�*�
w
discriminator_loss*a	   @NA?   @NA?      �?!   @NA?) ��f_��>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � b��       �E��	�<����A�*�
w
discriminator_loss*a	    EL?    EL?      �?!    EL?) ��@���>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        DσX�       �E��	�����A�*�
w
discriminator_loss*a	   �#@V?   �#@V?      �?!   �#@V?)@pP�c�>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        qp#��       �E��	a4�����A�*�
w
discriminator_loss*a	   �?[9?   �?[9?      �?!   �?[9?)� ϗ�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        3�u�       �E��	i����A�*�
w
discriminator_loss*a	    �<?    �<?      �?!    �<?) B�=���>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        P���       �E��	�g����A�*�
w
discriminator_loss*a	   @}�A?   @}�A?      �?!   @}�A?) y�0�>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        'Ur�       �E��	 �����A�*�
w
discriminator_loss*a	   `�1?   `�1?      �?!   `�1?)@^Uߵ�s>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ŵ�2�       �E��	������A�*�
w
discriminator_loss*a	   ��g1?   ��g1?      �?!   ��g1?) �@?+�r>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �I^j�       �E��	Sm����A�*�
w
discriminator_loss*a	   `��M?   `��M?      �?!   `��M?) ���}�>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Ɍ��       �E��	�<�����A�*�
w
discriminator_loss*a	   `p�<?   `p�<?      �?!   `p�<?) ������>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        u.��       �E��	7�$����A�*�
w
discriminator_loss*a	    t?    t?      �?!    t?) hs�rN>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	q�����A�*�
w
discriminator_loss*a	    %q0?    %q0?      �?!    %q0?) ��j�p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��ź�       M�	S�����A*�
w
discriminator_loss*a	   ���8?   ���8?      �?!   ���8?)��#N؂>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       ���g	$����A(*�
w
discriminator_loss*a	    zE;?    zE;?      �?!    zE;?) |��=�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        4,�A�       ���g	������AP*�
w
discriminator_loss*a	    n?    n?      �?!    n?) �m�u�7>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �.���       ���g	"������Ax*�
w
discriminator_loss*a	   `#w ?   `#w ?      �?!   `#w ?)@6�޽�P>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	��@����A�*�
w
discriminator_loss*a	   ��E?   ��E?      �?!   ��E?) $���>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        6�9g�       �E��	S֫����A�*�
w
discriminator_loss*a	   @�%?   @�%?      �?!   @�%?) q�:�,^>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���[�       �E��	������A�*�
w
discriminator_loss*a	   �D�/?   �D�/?      �?!   �D�/?)���m�o>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �1��       �E��	�������A�*�
w
discriminator_loss*a	   @{�&?   @{�&?      �?!   @{�&?) iM���_>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        t,:��       �E��	�����A�*�
w
discriminator_loss*a	   `GZ?   `GZ?      �?!   `GZ?) 3$WaG>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        `���       �E��	��l����A�*�
w
discriminator_loss*a	    �	,?    �	,?      �?!    �	,?) H���h>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        i����       �E��	�	�����A�*�
w
discriminator_loss*a	    �)?    �)?      �?!    �)?) � �Qd>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        A�U�       �E��	u
e����A�*�
w
discriminator_loss*a	   @=!?   @=!?      �?!   @=!?) y�N{PR>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�b�����A�*�
w
discriminator_loss*a	   �3Y0?   �3Y0?      �?!   �3Y0?) ĕMX�p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��s2�       �E��	<M����A�*�
w
discriminator_loss*a	   @�?   @�?      �?!   @�?)�X�h��/>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        # =��       �E��	�������A�*�
w
discriminator_loss*a	    S ?    S ?      �?!    S ?)@����P>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�*;����A�*�
w
discriminator_loss*a	    ?    ?      �?!    ?) @xD��;>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ^sC=�       �E��	�����A�*�
w
discriminator_loss*a	   �C9?   �C9?      �?!   �C9?)@��2Ts0>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �/��       �E��	�7����A�*�
w
discriminator_loss*a	   ��1-?   ��1-?      �?!   ��1-?)�0&�(�j>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��Q�       �E��	�'�����A�*�
w
discriminator_loss*a	   �v7?   �v7?      �?!   �v7?) ҆����>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ɱ� �       �E��	vD7����A�*�
w
discriminator_loss*a	   ��>?   ��>?      �?!   ��>?)���u��@>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �V���       �E��	������A�*�
w
discriminator_loss*a	   ��� ?   ��� ?      �?!   ��� ?) a��o�Q>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ׸���       �E��	R�=����A�*�
w
discriminator_loss*a	   ��0?   ��0?      �?!   ��0?)@�5�7>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �X�~�       M�	������A*�
w
discriminator_loss*a	   �dX?   �dX?      �?!   �dX?) �����,>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��/��       ���g	�3����A(*�
w
discriminator_loss*a	   `�X
?   `�X
?      �?!   `�X
?) �5Fr�%>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��ߢ�       ���g	$�����AP*�
w
discriminator_loss*a	   @�?   @�?      �?!   @�?)�t����K>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �TM��       ���g	��M����Ax*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) ��36(>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Zz6��       �E��	�7�����A�*�
w
discriminator_loss*a	    �J	?    �J	?      �?!    �J	?) $	��#>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        I�b�       �E��	�d����A�*�
w
discriminator_loss*a	   �S�1?   �S�1?      �?!   �S�1?)@=��:s>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        $�[�       �E��	1�����A�*�
w
discriminator_loss*a	    T)	?    T)	?      �?!    T)	?) ����#>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�f� ���A�*�
w
discriminator_loss*a	   �0�?   �0�?      �?!   �0�?) S���1>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �߀q�       �E��	Z����A�*�
w
discriminator_loss*a	    8�?    8�?      �?!    8�?) pXh�!>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �+�x�       �E��	�
����A�*�
w
discriminator_loss*a	   �h�?   �h�?      �?!   �h�?)���$�/>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���o�       �E��	t�=���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@J�*v>>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %xu�       �E��	������A�*�
w
discriminator_loss*a	   �f|?   �f|?      �?!   �f|?) R��ЛG>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �R�k�       �E��	l΂���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) $B{�3>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        b(��       �E��	�%���A�*�
w
discriminator_loss*a	   @�)?   @�)?      �?!   @�)?)�h!�c>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�k����A�*�
w
discriminator_loss*a	    �E!?    �E!?      �?!    �E!?) @Ru�R>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	m�R���A�*�
w
discriminator_loss*a	   `�~?   `�~?      �?!   `�~?) �s�-PD>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        H`��       �E��	@p����A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) �'�|4>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        S5o��       �E��	������A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) bu/�	>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ԩ�       �E��	��"���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@�m�m�>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��^�       �E��	������A�*�
w
discriminator_loss*a	    p|�>    p|�>      �?!    p|�>)  ��[	>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �:���       �E��	Öh���A�*�
w
discriminator_loss*a	    K	?    K	?      �?!    K	?) H
��#>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��+i�       �E��	��8	���A�*�
w
discriminator_loss*a	   ��E?   ��E?      �?!   ��E?)�TUN�� >26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        `Tr�       M�	�y�	���A*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)����ר>2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        g[�F�       ���g	�$�
���A(*�
w
discriminator_loss*a	   �<��>   �<��>      �?!   �<��>) ��յ�=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��G�       ���g	�l���AP*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) 7�����=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        .��R�       ���g	��+���Ax*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) ��
�>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        TbX�       �E��	�[����A�*�
w
discriminator_loss*a	   ��*?   ��*?      �?!   ��*?) �d x@e>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��,�       �E��	�t����A�*�
w
discriminator_loss*a	   ��(?   ��(?      �?!   ��(?) ��Gu\c>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        mg	y�       �E��	дp���A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>) Q��G��=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        -0iO�       �E��	�&���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �b_+�	>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        )�\��       �E��	p6����A�*�
w
discriminator_loss*a	   �fS�>   �fS�>      �?!   �fS�>)��Mrp�>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��Z�       �E��	6�����A�*�
w
discriminator_loss*a	   `��#?   `��#?      �?!   `��#?)@6�
9�X>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ~x���       �E��	�E���A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)���qJ�!>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %�γ�       �E��	�,���A�*�
w
discriminator_loss*a	   @h^?   @h^?      �?!   @h^?) A��dr7>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Q�2��       �E��	�%����A�*�
w
discriminator_loss*a	    U�?    U�?      �?!    U�?) r~�	jK>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �P1�       �E��	�ڱ���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��Ӽ~�=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        4R�       �E��	_����A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) q���8>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?         �X�       �E��	3�Z���A�*�
w
discriminator_loss*a	    wB?    wB?      �?!    wB?) 5��.7>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��%�       �E��	��& ���A�*�
w
discriminator_loss*a	   ��N�>   ��N�>      �?!   ��N�>) rՈ��>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        A�#��       �E��	lK "���A�*�
w
discriminator_loss*a	   �$?   �$?      �?!   �$?)����cL>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        n��       �E��	w�"���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) dĻǊ>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �E�_�       �E��	h��$���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �;���=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        eWb��       �E��	��&���A�*�
w
discriminator_loss*a	   `8{�>   `8{�>      �?!   `8{�>) Qŧ0)�=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��	-�       �E��	o��(���A�*�
w
discriminator_loss*a	    �i�>    �i�>      �?!    �i�>) �'�e�=2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��(D�       M�	�oM*���A*�
w
discriminator_loss*a	    5D?    5D?      �?!    5D?) �/� 3>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       ���g	��K+���A(*�
w
discriminator_loss*a	    �z�>    �z�>      �?!    �z�>) t��*(�=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �%�n�       ���g	��-���AP*�
w
discriminator_loss*a	   ��o�>   ��o�>      �?!   ��o�>) ����)�=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        o_6�       ���g	>�-���Ax*�
w
discriminator_loss*a	   �y��>   �y��>      �?!   �y��>) ,^!$�=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        J�+t�       �E��	1��.���A�*�
w
discriminator_loss*a	    7J�>    7J�>      �?!    7J�>) }��=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��D2�       �E��	��/���A�*�
w
discriminator_loss*a	   ��z�>   ��z�>      �?!   ��z�>) �K���=2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        R�       �E��	���0���A�*�
w
discriminator_loss*a	   ��N�>   ��N�>      �?!   ��N�>)@4�	M�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��S�       �E��	��1���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ��j
�=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Z��Y�       �E��	լ�2���A�*�
w
discriminator_loss*a	   ��f�>   ��f�>      �?!   ��f�>) ŐiQ��=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        T2�$�       �E��	qm3���A�*�
w
discriminator_loss*a	    7��>    7��>      �?!    7��>)@�M��m�=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        |�;�       �E��	|K4���A�*�
w
discriminator_loss*a	    �t�>    �t�>      �?!    �t�>) �m�>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Щ� �       �E��	U35���A�*�
w
discriminator_loss*a	   @`*?   @`*?      �?!   @`*?) ��`j>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Hn�       �E��	>�A6���A�*�
w
discriminator_loss*a	    �|�>    �|�>      �?!    �|�>) �nd�<�=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        (
��       �E��	T7���A�*�
w
discriminator_loss*a	   @n�>   @n�>      �?!   @n�>) �OK�=2�ѩ�-�>���%�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���G�       �E��	ܻF8���A�*�
w
discriminator_loss*a	   ��u�>   ��u�>      �?!   ��u�>) ġ+*�=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��|��       �E��	��F9���A�*�
w
discriminator_loss*a	   �W�>   �W�>      �?!   �W�>) ������=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��j'�       �E��	�::���A�*�
w
discriminator_loss*a	   �(F�>   �(F�>      �?!   �(F�>)@&�[#�=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���9�       �E��	��B;���A�*�
w
discriminator_loss*a	   �ș�>   �ș�>      �?!   �ș�>)@�ǒ�=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �T��       �E��	�L<���A�*�
w
discriminator_loss*a	   �%��>   �%��>      �?!   �%��>) �k�wի=2['�?��>K+�E���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �b��       �E��	��N=���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) \!��>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        :a5�       �E��	�iB>���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)��o, F>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ^��{�       �E��	�D?���A�*�
w
discriminator_loss*a	   �@F�>   �@F�>      �?!   �@F�>) �'?>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �IS��       M�	�C@���A*�
w
discriminator_loss*a	   �R�>   �R�>      �?!   �R�>) �w��� >2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �V�M�       ���g	�CA���A(*�
w
discriminator_loss*a	   @�L�>   @�L�>      �?!   @�L�>)��b�R�=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        h�=.�       ���g	�=B���AP*�
w
discriminator_loss*a	   �~%�>   �~%�>      �?!   �~%�>) $���=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        (��8�       ���g	]FC���Ax*�
w
discriminator_loss*a	   @A`�>   @A`�>      �?!   @A`�>)���k�=2�iD*L��>E��a�W�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        }�
D�       �E��	E�ZD���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) 	OK�E�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	j�kE���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�.�Bė=2��~���>�XQ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���w�       �E��	�tF���A�*�
w
discriminator_loss*a	   `�	�>   `�	�>      �?!   `�	�>)@.X���=2�ѩ�-�>���%�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��On�       �E��	?�qG���A�*�
w
discriminator_loss*a	   �6��>   �6��>      �?!   �6��>)���`���=2�����>
�/eq
�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ٜ���       �E��	��hH���A�*�
w
discriminator_loss*a	    #^�>    #^�>      �?!    #^�>) ��[���=2��~]�[�>��>M|K�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �w���       �E��	�jI���A�*�
w
discriminator_loss*a	   �e��>   �e��>      �?!   �e��>)@z�-���=2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        6]ޒ�       �E��	M�fJ���A�*�
w
discriminator_loss*a	   �ZK�>   �ZK�>      �?!   �ZK�>) �¹q�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � v�       �E��	acK���A�*�
w
discriminator_loss*a	   `�X�>   `�X�>      �?!   `�X�>)@�O���=2�ѩ�-�>���%�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        )���       �E��	�@bL���A�*�
w
discriminator_loss*a	   @+�>   @+�>      �?!   @+�>)�t�wd�>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��N�       �E��	psM���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@&b��g�=2K+�E���>jqs&\��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � ���       �E��	O�N���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ]D��=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Z�O��       �E��	��O���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) @��#��=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #�Ҭ�       �E��	���P���A�*�
w
discriminator_loss*a	   �=�>   �=�>      �?!   �=�>)�(
7�=2['�?��>K+�E���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        j�%v�       �E��	�Q���A�*�
w
discriminator_loss*a	   �?p�>   �?p�>      �?!   �?p�>) ����=2E��a�W�>�ѩ�-�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �X�       �E��	�x�R���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) a	��z=2�u`P+d�>0�6�/n�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��X�       �E��	ET���A�*�
w
discriminator_loss*a	    Kw�>    Kw�>      �?!    Kw�>)@��v�O�=2jqs&\��>��~]�[�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��o��       �E��	�)U���A�*�
w
discriminator_loss*a	    {��>    {��>      �?!    {��>)@|S�!��=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        tL$W�       �E��	;V���A�*�
w
discriminator_loss*a	   @�?   @�?      �?!   @�?)���e�`/>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Or��       M�	��4W���A*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �
ó=2jqs&\��>��~]�[�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        5iU8�       ���g	1OX���A(*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) 	���1>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ^�<��       ���g	�jxY���AP*�
w
discriminator_loss*a	    �X�>    �X�>      �?!    �X�>) @^[T{�=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �-���       ���g	�4�Z���Ax*�
w
discriminator_loss*a	    �>    �>      �?!    �>) @ D2R=2�u��gr�>�MZ��K�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Cy��       �E��	�-�[���A�*�
w
discriminator_loss*a	    *��>    *��>      �?!    *��>) @n���=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �e��       �E��	Y]���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) i� ���=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��Px�       �E��	�.^���A�*�
w
discriminator_loss*a	   @rg�>   @rg�>      �?!   @rg�>) �'�(��=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        H:4��       �E��	�VW_���A�*�
w
discriminator_loss*a	   @ �>   @ �>      �?!   @ �>)�� p�h=2���?�ګ>����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �|���       �E��	aa�`���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@.]$�}=2�u`P+d�>0�6�/n�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        _����       �E��	���a���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)��i�2�=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��?�       �E��	1M�b���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �%�UG�=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �f��       �E��	�&Pd���A�*�
w
discriminator_loss*a	    #H�>    #H�>      �?!    #H�>) �6N�l�=20�6�/n�>5�"�g��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ~<�L�       �E��	2;~e���A�*�
w
discriminator_loss*a	    6h�>    6h�>      �?!    6h�>) �+z5�=2['�?��>K+�E���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %����       �E��	�L�f���A�*�
w
discriminator_loss*a	   @�e�>   @�e�>      �?!   @�e�>)���)(�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �vg��       �E��	�h���A�*�
w
discriminator_loss*a	   @ �>   @ �>      �?!   @ �>) �
^�t=2豪}0ڰ>��n����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        R���       �E��	]�i���A�*�
w
discriminator_loss*a	    ^�>    ^�>      �?!    ^�>)@��"��=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �f���       �E��	[��j���A�*�
w
discriminator_loss*a	   �ě�>   �ě�>      �?!   �ě�>) D�xu=�=2�ѩ�-�>���%�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        bd��       �E��	,�Il���A�*�
w
discriminator_loss*a	   �U��>   �U��>      �?!   �U��>)@:����=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	
w�m���A�*�
w
discriminator_loss*a	    Bx�>    Bx�>      �?!    Bx�>)  ��&��=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���\�       �E��	]l�n���A�*�
w
discriminator_loss*a	   �:��>   �:��>      �?!   �:��>) g˵_��=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �F��       �E��	(Wdp���A�*�
w
discriminator_loss*a	   ��6�>   ��6�>      �?!   ��6�>) $D˄�=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �7�       �E��	��q���A�*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@V����=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       M�	L0s���A*�
w
discriminator_loss*a	    ?� ?    ?� ?      �?!    ?� ?) �4��>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        qVv�       ���g	�V�t���A(*�
w
discriminator_loss*a	   `52�>   `52�>      �?!   `52�>)@JNe�=2�ѩ�-�>���%�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        I
�;�       ���g	���u���AP*�
w
discriminator_loss*a	   �>�>   �>�>      �?!   �>�>)@�>� �=2K+�E���>jqs&\��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       ���g	_	Cw���Ax*�
w
discriminator_loss*a	   ��O�>   ��O�>      �?!   ��O�>)�l&_��=2E��a�W�>�ѩ�-�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ;��t�       �E��	|D�x���A�*�
w
discriminator_loss*a	   `T�>   `T�>      �?!   `T�>)@"����=2��~���>�XQ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	��z���A�*�
w
discriminator_loss*a	     �>     �>      �?!     �>)  K�^=2���]���>�5�L�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���@�       �E��	�݀{���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ȳTҶ>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �:��       �E��	��|���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  H�>O�=2�*��ڽ>�[�=�k�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Y��6�       �E��	�
~���A�*�
w
discriminator_loss*a	    (��>    (��>      �?!    (��>) P"��|�=2�����>
�/eq
�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	)[���A�*�
w
discriminator_loss*a	   @"v�>   @"v�>      �?!   @"v�>) Qy���=2K+�E���>jqs&\��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]N��       �E��	
������A�*�
w
discriminator_loss*a	    H�>    H�>      �?!    H�>) H��O��=2
�/eq
�>;�"�q�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��l�       �E��	&�����A�*�
w
discriminator_loss*a	   @[�>   @[�>      �?!   @[�>) i�e�=2��~]�[�>��>M|K�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �-�       �E��	E�����A�*�
w
discriminator_loss*a	   @&��>   @&��>      �?!   @&��>)���<��=2�����>
�/eq
�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �QN��       �E��	$s�����A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) �K�h=2���?�ګ>����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���6�       �E��	9\�����A�*�
w
discriminator_loss*a	   ��s�>   ��s�>      �?!   ��s�>) r����=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �4D��       �E��	�$����A�*�
w
discriminator_loss*a	    di�>    di�>      �?!    di�>)@�����=2K+�E���>jqs&\��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �[�       �E��	��~����A�*�
w
discriminator_loss*a	    �W?    �W?      �?!    �W?) ���_�>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �r3f�       �E��	�uߊ���A�*�
w
discriminator_loss*a	    Xk�>    Xk�>      �?!    Xk�>)  �*���=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        N�8�       �E��	V1~����A�*�
w
discriminator_loss*a	   @T�>   @T�>      �?!   @T�>) ˳�ә=2�XQ��>�����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        bG`��       �E��	D�.����A�*�
w
discriminator_loss*a	    `�>    `�>      �?!    `�>) b��3��=20�6�/n�>5�"�g��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��.f�       �E��	6����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �	���o=2����>豪}0ڰ>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        -�}�       �E��	������A�*�
w
discriminator_loss*a	   �Т>   �Т>      �?!   �Т>) ���V=2�MZ��K�>��|�~�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        [xsI�       M�	�����A*�
w
discriminator_loss*a	    p�>    p�>      �?!    p�>) �L��n=2����>豪}0ڰ>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        0�0��       ���g		�l����A(*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �Oܑ
�=20�6�/n�>5�"�g��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       ���g	E�����AP*�
w
discriminator_loss*a	    #�>    #�>      �?!    #�>) ��ZH�=2K+�E���>jqs&\��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        .��+�       ���g	��@����Ax*�
w
discriminator_loss*a	   �3�>   �3�>      �?!   �3�>) $\㭃=25�"�g��>G&�$�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �M���       �E��	(E�����A�*�
w
discriminator_loss*a	   ��B�>   ��B�>      �?!   ��B�>) �v�؝�=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        E.t�       �E��	������A�*�
w
discriminator_loss*a	   �:��>   �:��>      �?!   �:��>) �j�~�=2
�/eq
�>;�"�q�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	��~����A�*�
w
discriminator_loss*a	   �`�>   �`�>      �?!   �`�>)@���E�P=2�u��gr�>�MZ��K�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	]������A�*�
w
discriminator_loss*a	   @0�>   @0�>      �?!   @0�>) q�5��t=2豪}0ڰ>��n����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        M g#�       �E��	�a�����A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  \��=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �f�       �E��	�N����A�*�
w
discriminator_loss*a	   �00�>   �00�>      �?!   �00�>) �)����=2G&�$�>�*��ڽ>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Q7�       �E��	dSΡ���A�*�
w
discriminator_loss*a	   `:��>   `:��>      �?!   `:��>)@�<"���=2��~]�[�>��>M|K�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �GN�       �E��	oVD����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��xM�=2��~���>�XQ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        M�