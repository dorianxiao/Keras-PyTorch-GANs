       �K"	  @����Abrain.Event:2��ؕ�	     �g�	�1b����A"��
u
Generator/noise_inPlaceholder*
dtype0*'
_output_shapes
:���������d*
shape:���������d
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
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
: 
�
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
_output_shapes
:	d�*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
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
3Generator/first_layer/fully_connected/kernel/AssignAssign,Generator/first_layer/fully_connected/kernelGGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(
�
1Generator/first_layer/fully_connected/kernel/readIdentity,Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
�
<Generator/first_layer/fully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB�*    
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
&Generator/first_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
$Generator/first_layer/leaky_relu/mulMul&Generator/first_layer/leaky_relu/alpha-Generator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
 Generator/first_layer/leaky_reluMaximum$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
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
seed2 *
dtype0* 
_output_shapes
:
��*

seed *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
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
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
�
4Generator/second_layer/fully_connected/kernel/AssignAssign-Generator/second_layer/fully_connected/kernelHGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
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
0Generator/second_layer/fully_connected/bias/readIdentity+Generator/second_layer/fully_connected/bias*
_output_shapes	
:�*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias
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
AGenerator/second_layer/batch_normalization/gamma/Initializer/onesConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
0Generator/second_layer/batch_normalization/gamma
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container 
�
7Generator/second_layer/batch_normalization/gamma/AssignAssign0Generator/second_layer/batch_normalization/gammaAGenerator/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
	container *
shape:�
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
KGenerator/second_layer/batch_normalization/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:�*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
valueB�*  �?
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
:Generator/second_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
8Generator/second_layer/batch_normalization/batchnorm/addAdd?Generator/second_layer/batch_normalization/moving_variance/read:Generator/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:�
�
:Generator/second_layer/batch_normalization/batchnorm/RsqrtRsqrt8Generator/second_layer/batch_normalization/batchnorm/add*
_output_shapes	
:�*
T0
�
8Generator/second_layer/batch_normalization/batchnorm/mulMul:Generator/second_layer/batch_normalization/batchnorm/Rsqrt5Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
:Generator/second_layer/batch_normalization/batchnorm/mul_1Mul.Generator/second_layer/fully_connected/BiasAdd8Generator/second_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
�
:Generator/second_layer/batch_normalization/batchnorm/mul_2Mul;Generator/second_layer/batch_normalization/moving_mean/read8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
8Generator/second_layer/batch_normalization/batchnorm/subSub4Generator/second_layer/batch_normalization/beta/read:Generator/second_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:�*
T0
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
!Generator/second_layer/leaky_reluMaximum%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
MGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      
�
KGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *���
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
dtype0* 
_output_shapes
:
��*

seed *
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
seed2 
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
GGenerator/third_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��
�
,Generator/third_layer/fully_connected/kernel
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
3Generator/third_layer/fully_connected/kernel/AssignAssign,Generator/third_layer/fully_connected/kernelGGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container 
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean*
	container 
�
<Generator/third_layer/batch_normalization/moving_mean/AssignAssign5Generator/third_layer/batch_normalization/moving_meanGGenerator/third_layer/batch_normalization/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean
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
VariableV2*
shared_name *L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
>Generator/third_layer/batch_normalization/moving_variance/readIdentity9Generator/third_layer/batch_normalization/moving_variance*
_output_shapes	
:�*
T0*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance
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
9Generator/third_layer/batch_normalization/batchnorm/RsqrtRsqrt7Generator/third_layer/batch_normalization/batchnorm/add*
_output_shapes	
:�*
T0
�
7Generator/third_layer/batch_normalization/batchnorm/mulMul9Generator/third_layer/batch_normalization/batchnorm/Rsqrt4Generator/third_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
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
7Generator/third_layer/batch_normalization/batchnorm/subSub3Generator/third_layer/batch_normalization/beta/read9Generator/third_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
9Generator/third_layer/batch_normalization/batchnorm/add_1Add9Generator/third_layer/batch_normalization/batchnorm/mul_17Generator/third_layer/batch_normalization/batchnorm/sub*
T0*(
_output_shapes
:����������
k
&Generator/third_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
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
LGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      
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
TGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
��*

seed 
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/subSubJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
_output_shapes
: 
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/mulMulTGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/sub*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��
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
VariableV2*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
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
+Generator/last_layer/fully_connected/MatMulMatMul Generator/third_layer/leaky_relu0Generator/last_layer/fully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
,Generator/last_layer/fully_connected/BiasAddBiasAdd+Generator/last_layer/fully_connected/MatMul.Generator/last_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
OGenerator/last_layer/batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:�
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
5Generator/last_layer/batch_normalization/gamma/AssignAssign.Generator/last_layer/batch_normalization/gamma?Generator/last_layer/batch_normalization/gamma/Initializer/ones*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container 
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
VGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB:�*
dtype0*
_output_shapes
:
�
LGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
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
YGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB:�*
dtype0*
_output_shapes
:
�
OGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
valueB
 *  �?
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
?Generator/last_layer/batch_normalization/moving_variance/AssignAssign8Generator/last_layer/batch_normalization/moving_varianceIGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
8Generator/last_layer/batch_normalization/batchnorm/RsqrtRsqrt6Generator/last_layer/batch_normalization/batchnorm/add*
_output_shapes	
:�*
T0
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
8Generator/last_layer/batch_normalization/batchnorm/mul_2Mul9Generator/last_layer/batch_normalization/moving_mean/read6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
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
Generator/last_layer/leaky_reluMaximum#Generator/last_layer/leaky_relu/mul8Generator/last_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
<Generator/fake_image/kernel/Initializer/random_uniform/shapeConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
:Generator/fake_image/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *z�k�
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
:Generator/fake_image/kernel/Initializer/random_uniform/mulMulDGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniform:Generator/fake_image/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��
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
VariableV2*.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
"Generator/fake_image/kernel/AssignAssignGenerator/fake_image/kernel6Generator/fake_image/kernel/Initializer/random_uniform*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
 Generator/fake_image/kernel/readIdentityGenerator/fake_image/kernel*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��
�
+Generator/fake_image/bias/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Generator/fake_image/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:�
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
Generator/fake_image/MatMulMatMulGenerator/last_layer/leaky_relu Generator/fake_image/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
Generator/fake_image/BiasAddBiasAddGenerator/fake_image/MatMulGenerator/fake_image/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
r
Generator/fake_image/TanhTanhGenerator/fake_image/BiasAdd*
T0*(
_output_shapes
:����������
z
Discriminator/real_inPlaceholder*
shape:����������*
dtype0*(
_output_shapes
:����������
�
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
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
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *HY�=
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
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
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
7Discriminator/first_layer/fully_connected/kernel/AssignAssign0Discriminator/first_layer/fully_connected/kernelKDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
6Discriminator/second_layer/fully_connected/kernel/readIdentity1Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
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
1Discriminator/second_layer/fully_connected/MatMulMatMul$Discriminator/first_layer/leaky_relu6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
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
8Discriminator/prob/kernel/Initializer/random_uniform/subSub8Discriminator/prob/kernel/Initializer/random_uniform/max8Discriminator/prob/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*,
_class"
 loc:@Discriminator/prob/kernel
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
VariableV2*,
_class"
 loc:@Discriminator/prob/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name 
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
Discriminator/prob/bias/readIdentityDiscriminator/prob/bias*
_output_shapes
:*
T0**
_class 
loc:@Discriminator/prob/bias
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
,Discriminator/first_layer_1/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
*Discriminator/first_layer_1/leaky_relu/mulMul,Discriminator/first_layer_1/leaky_relu/alpha3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
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
logistic_loss/SelectSelectlogistic_loss/GreaterEqualDiscriminator/prob/BiasAddlogistic_loss/zeros_like*'
_output_shapes
:���������*
T0
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
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*'
_output_shapes
:���������
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*'
_output_shapes
:���������*
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

zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
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
logistic_loss_1/NegNegDiscriminator/prob_1/BiasAdd*'
_output_shapes
:���������*
T0
�
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
v
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAdd
zeros_like*
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
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:���������*
T0
�
logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:���������
�
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:���������
j
logistic_loss_2/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
w
logistic_loss_2/mulMulDiscriminator/prob_1/BiasAddones_like_1*
T0*'
_output_shapes
:���������
y
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*
T0*'
_output_shapes
:���������
f
logistic_loss_2/ExpExplogistic_loss_2/Select_1*'
_output_shapes
:���������*
T0
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
Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
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
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
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
�
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
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:���������*
T0
t
#gradients/Mean_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
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
�
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
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
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
�
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
�
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*'
_output_shapes
:���������
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
�
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
�
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*'
_output_shapes
:���������
�
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*'
_output_shapes
:���������*
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
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1*'
_output_shapes
:���������
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
�
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
_output_shapes
:*
T0
�
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
�
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*'
_output_shapes
:���������
�
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1
�
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:���������
�
-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*'
_output_shapes
:���������*
T0
�
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:���������*
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
�
8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/sub_grad/Shape*gradients/logistic_loss_1/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients/logistic_loss_1/sub_grad/SumSum7gradients/logistic_loss_1_grad/tuple/control_dependency8gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
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
�
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:���������
�
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
�
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:���������
�
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:���������
�
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
�
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*'
_output_shapes
:���������*
T0
�
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&gradients/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
�
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*'
_output_shapes
:���������
�
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:���������
�
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
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
�
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1
zeros_like*'
_output_shapes
:���������*
T0
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
(gradients/logistic_loss_1/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
=gradients/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity,gradients/logistic_loss_1/mul_grad/Reshape_14^gradients/logistic_loss_1/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients/logistic_loss_1/mul_grad/Reshape_1
�
&gradients/logistic_loss_1/Exp_grad/mulMul(gradients/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0*'
_output_shapes
:���������
�
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:���������
�
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0*'
_output_shapes
:���������
�
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:���������
�
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
�
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select
�
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1
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
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1
�
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:���������
�
5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
data_formatNHWC*
_output_shapes
:*
T0
�
:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN6^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad
�
Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:���������
�
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
gradients/AddN_1AddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N*'
_output_shapes
:���������
�
7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
T0*
data_formatNHWC*
_output_shapes
:
�
<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_18^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*'
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
�
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
gradients/AddN_2AddNDgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
:*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad
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
Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosFill>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Bgradients/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
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
=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency<gradients/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1Reshape<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
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
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_3AddNCgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
N*
_output_shapes
:	�
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
<gradients/Discriminator/second_layer/leaky_relu/mul_grad/SumSum<gradients/Discriminator/second_layer/leaky_relu/mul_grad/MulNgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1L^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
�
gradients/AddN_4AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
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
gradients/AddN_5AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityIgradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1R^gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
Igradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulT^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul
�
]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_6AddN\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
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
@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/Discriminator/first_layer/leaky_relu_grad/SelectSelect@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency9gradients/Discriminator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
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
Lgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependencyIdentity;gradients/Discriminator/first_layer/leaky_relu_grad/ReshapeE^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*N
_classD
B@loc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape
�
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
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
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
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
Bgradients/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual*Discriminator/first_layer_1/leaky_relu/mul3Discriminator/first_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
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
;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
�
gradients/AddN_7AddN[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
��
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
=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum_1Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Mul_1Ogradients/Discriminator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1IdentityAgradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1I^gradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer/leaky_relu/mul_grad/Reshape_1
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
Ogradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulMulNgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency3Discriminator/first_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum=gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/MulOgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
�
gradients/AddN_8AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
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
gradients/AddN_9AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/real_inYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
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
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Jgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/fake_image/Tanh[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
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
gradients/AddN_10AddN[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes	
:�*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
gradients/AddN_11AddNZgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
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
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
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
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *w�?
�
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
GDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillWDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorMDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0
�
5Discriminator/first_layer/fully_connected/kernel/Adam
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
<Discriminator/first_layer/fully_connected/kernel/Adam/AssignAssign5Discriminator/first_layer/fully_connected/kernel/AdamGDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
>Discriminator/first_layer/fully_connected/kernel/Adam_1/AssignAssign7Discriminator/first_layer/fully_connected/kernel/Adam_1IDiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
�
<Discriminator/first_layer/fully_connected/kernel/Adam_1/readIdentity7Discriminator/first_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container 
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
VariableV2*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
?Discriminator/second_layer/fully_connected/kernel/Adam_1/AssignAssign8Discriminator/second_layer/fully_connected/kernel/Adam_1JDiscriminator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
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
VariableV2*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
;Discriminator/second_layer/fully_connected/bias/Adam/AssignAssign4Discriminator/second_layer/fully_connected/bias/AdamFDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
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
%Discriminator/prob/kernel/Adam/AssignAssignDiscriminator/prob/kernel/Adam0Discriminator/prob/kernel/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�
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
%Discriminator/prob/kernel/Adam_1/readIdentity Discriminator/prob/kernel/Adam_1*
_output_shapes
:	�*
T0*,
_class"
 loc:@Discriminator/prob/kernel
�
.Discriminator/prob/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    
�
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
0Discriminator/prob/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:**
_class 
loc:@Discriminator/prob/bias*
valueB*    
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
Adam/beta2Adam/epsilongradients/AddN_11*
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
Adam/beta2Adam/epsilongradients/AddN_10*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
GAdam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam1Discriminator/second_layer/fully_connected/kernel6Discriminator/second_layer/fully_connected/kernel/Adam8Discriminator/second_layer/fully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
�
EAdam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/bias4Discriminator/second_layer/fully_connected/bias/Adam6Discriminator/second_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
�
/Adam/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernelDiscriminator/prob/kernel/Adam Discriminator/prob/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
use_nesterov( *
_output_shapes
:	�
�
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
�
Adam/mulMulbeta1_power/read
Adam/beta1E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
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
Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
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
 *  �?
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
v
%gradients_1/Mean_2_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
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
�
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*'
_output_shapes
:���������*

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
gradients_1/Mean_2_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_1/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
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
�
gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
�
 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
�
gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*
T0*'
_output_shapes
:���������
y
&gradients_1/logistic_loss_2_grad/ShapeShapelogistic_loss_2/sub*
T0*
out_type0*
_output_shapes
:
}
(gradients_1/logistic_loss_2_grad/Shape_1Shapelogistic_loss_2/Log1p*
T0*
out_type0*
_output_shapes
:
�
6gradients_1/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_2_grad/Shape(gradients_1/logistic_loss_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients_1/logistic_loss_2_grad/SumSumgradients_1/Mean_2_grad/truediv6gradients_1/logistic_loss_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients_1/logistic_loss_2_grad/ReshapeReshape$gradients_1/logistic_loss_2_grad/Sum&gradients_1/logistic_loss_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&gradients_1/logistic_loss_2_grad/Sum_1Sumgradients_1/Mean_2_grad/truediv8gradients_1/logistic_loss_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*gradients_1/logistic_loss_2_grad/Reshape_1Reshape&gradients_1/logistic_loss_2_grad/Sum_1(gradients_1/logistic_loss_2_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
1gradients_1/logistic_loss_2_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_2_grad/Reshape+^gradients_1/logistic_loss_2_grad/Reshape_1
�
9gradients_1/logistic_loss_2_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_2_grad/Reshape2^gradients_1/logistic_loss_2_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_2_grad/Reshape
�
;gradients_1/logistic_loss_2_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_2_grad/Reshape_12^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss_2_grad/Reshape_1*'
_output_shapes
:���������
�
*gradients_1/logistic_loss_2/sub_grad/ShapeShapelogistic_loss_2/Select*
T0*
out_type0*
_output_shapes
:

,gradients_1/logistic_loss_2/sub_grad/Shape_1Shapelogistic_loss_2/mul*
T0*
out_type0*
_output_shapes
:
�
:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/sub_grad/Shape,gradients_1/logistic_loss_2/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
(gradients_1/logistic_loss_2/sub_grad/SumSum9gradients_1/logistic_loss_2_grad/tuple/control_dependency:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
,gradients_1/logistic_loss_2/sub_grad/ReshapeReshape(gradients_1/logistic_loss_2/sub_grad/Sum*gradients_1/logistic_loss_2/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
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
�
.gradients_1/logistic_loss_2/sub_grad/Reshape_1Reshape(gradients_1/logistic_loss_2/sub_grad/Neg,gradients_1/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
5gradients_1/logistic_loss_2/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/sub_grad/Reshape/^gradients_1/logistic_loss_2/sub_grad/Reshape_1
�
=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/sub_grad/Reshape6^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/sub_grad/Reshape*'
_output_shapes
:���������
�
?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/sub_grad/Reshape_16^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/sub_grad/Reshape_1
�
,gradients_1/logistic_loss_2/Log1p_grad/add/xConst<^gradients_1/logistic_loss_2_grad/tuple/control_dependency_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
*gradients_1/logistic_loss_2/Log1p_grad/addAdd,gradients_1/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*
T0*'
_output_shapes
:���������
�
1gradients_1/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_2/Log1p_grad/add*
T0*'
_output_shapes
:���������
�
*gradients_1/logistic_loss_2/Log1p_grad/mulMul;gradients_1/logistic_loss_2_grad/tuple/control_dependency_11gradients_1/logistic_loss_2/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
2gradients_1/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:���������*
T0
�
.gradients_1/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_2/Select_grad/zeros_like*
T0*'
_output_shapes
:���������
�
0gradients_1/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients_1/logistic_loss_2/Select_grad/zeros_like=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
8gradients_1/logistic_loss_2/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_2/Select_grad/Select1^gradients_1/logistic_loss_2/Select_grad/Select_1
�
@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_2/Select_grad/Select9^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:���������
�
Bgradients_1/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_2/Select_grad/Select_19^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_grad/Select_1*'
_output_shapes
:���������
�
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
�
:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/mul_grad/Shape,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
(gradients_1/logistic_loss_2/mul_grad/MulMul?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1ones_like_1*
T0*'
_output_shapes
:���������
�
(gradients_1/logistic_loss_2/mul_grad/SumSum(gradients_1/logistic_loss_2/mul_grad/Mul:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
,gradients_1/logistic_loss_2/mul_grad/ReshapeReshape(gradients_1/logistic_loss_2/mul_grad/Sum*gradients_1/logistic_loss_2/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
*gradients_1/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
*gradients_1/logistic_loss_2/mul_grad/Sum_1Sum*gradients_1/logistic_loss_2/mul_grad/Mul_1<gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
.gradients_1/logistic_loss_2/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_2/mul_grad/Sum_1,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
5gradients_1/logistic_loss_2/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/mul_grad/Reshape/^gradients_1/logistic_loss_2/mul_grad/Reshape_1
�
=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/mul_grad/Reshape6^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/mul_grad/Reshape*'
_output_shapes
:���������
�
?gradients_1/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/mul_grad/Reshape_16^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:���������
�
(gradients_1/logistic_loss_2/Exp_grad/mulMul*gradients_1/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*
T0*'
_output_shapes
:���������
�
4gradients_1/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*
T0*'
_output_shapes
:���������
�
0gradients_1/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual(gradients_1/logistic_loss_2/Exp_grad/mul4gradients_1/logistic_loss_2/Select_1_grad/zeros_like*
T0*'
_output_shapes
:���������
�
2gradients_1/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual4gradients_1/logistic_loss_2/Select_1_grad/zeros_like(gradients_1/logistic_loss_2/Exp_grad/mul*
T0*'
_output_shapes
:���������
�
:gradients_1/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_2/Select_1_grad/Select3^gradients_1/logistic_loss_2/Select_1_grad/Select_1
�
Bgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_2/Select_1_grad/Select;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_1_grad/Select
�
Dgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_2/Select_1_grad/Select_1;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/logistic_loss_2/Select_1_grad/Select_1*'
_output_shapes
:���������
�
(gradients_1/logistic_loss_2/Neg_grad/NegNegBgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
gradients_1/AddNAddN@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_2/Neg_grad/Neg*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*
N*'
_output_shapes
:���������
�
9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes
:
�
>gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN:^gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
�
Fgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select
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
5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
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
Egradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Ngradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
?gradients_1/Discriminator/second_layer_1/leaky_relu_grad/SelectSelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Agradients_1/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectEgradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
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
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulRgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Tgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityFgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1N^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_1AddNSgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Qgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Vgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1R^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1W^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
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
Kgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Mgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
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
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Cgradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectSelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1SelectDgradients_1/Discriminator/first_layer_1/leaky_relu_grad/GreaterEqual=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/zeros]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SumSum>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectMgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Agradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
Rgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityAgradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1I^gradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
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
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Sgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Egradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Cgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Lgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpD^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeF^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
�
Tgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityCgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeM^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*V
_classL
JHloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape
�
Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityEgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1M^gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_2AddNRgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Pgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ugradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2Q^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2V^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
_gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradV^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Jgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
3gradients_1/Generator/fake_image/MatMul_grad/MatMulMatMulFgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency Generator/fake_image/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2ShapeEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
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
7gradients_1/Generator/last_layer/leaky_relu_grad/SelectSelect=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqualEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency6gradients_1/Generator/last_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Select=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqual6gradients_1/Generator/last_layer/leaky_relu_grad/zerosEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
4gradients_1/Generator/last_layer/leaky_relu_grad/SumSum7gradients_1/Generator/last_layer/leaky_relu_grad/SelectFgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Igradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeB^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
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
Jgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulMulIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/SumSum8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulJgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
gradients_1/AddN_3AddNKgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Generator/last_layer/batch_normalization/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
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
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_3_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_3agradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
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
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul,Generator/last_layer/fully_connected/BiasAddbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Igradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:�*
T0
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
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
�
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
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
Ugradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*V
_classL
JHloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul
�
Wgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_4AddNdgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_43Generator/last_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_48Generator/last_layer/batch_normalization/batchnorm/Rsqrt*
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
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
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
8gradients_1/Generator/third_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/third_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/third_layer/leaky_relu_grad/zerosUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
5gradients_1/Generator/third_layer/leaky_relu_grad/SumSum8gradients_1/Generator/third_layer/leaky_relu_grad/SelectGgradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/third_layer/leaky_relu_grad/Sum7gradients_1/Generator/third_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Igradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
=gradients_1/Generator/third_layer/leaky_relu/mul_grad/ReshapeReshape9gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
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
gradients_1/AddN_5AddNLgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
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
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_5`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
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
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_5bgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency7Generator/third_layer/batch_normalization/batchnorm/mul*(
_output_shapes
:����������*
T0
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
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/NegNegegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpf^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1M^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/Neg
�
agradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityegradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
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
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
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
Fgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1MatMul!Generator/second_layer/leaky_reluWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
�
Ngradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_depsNoOpE^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulG^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1
�
Vgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Xgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*Y
_classO
MKloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1
�
gradients_1/AddN_6AddNegradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_64Generator/third_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_69Generator/third_layer/batch_normalization/batchnorm/Rsqrt*
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
>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
9gradients_1/Generator/second_layer/leaky_relu_grad/SelectSelect?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency8gradients_1/Generator/second_layer/leaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Select?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqual8gradients_1/Generator/second_layer/leaky_relu_grad/zerosVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
6gradients_1/Generator/second_layer/leaky_relu_grad/SumSum9gradients_1/Generator/second_layer/leaky_relu_grad/SelectHgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeReshape6gradients_1/Generator/second_layer/leaky_relu_grad/Sum8gradients_1/Generator/second_layer/leaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1Sum;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Jgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1Reshape8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
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
<gradients_1/Generator/second_layer/leaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1Shape:Generator/second_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
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
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/SumSum:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulLgradients_1/Generator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Ogradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeH^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1H^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*S
_classI
GEloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
�
gradients_1/AddN_7AddNMgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
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
agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeSgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_7agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
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
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_7cgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
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
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul.Generator/second_layer/fully_connected/BiasAdddgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1cgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
�
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*d
_classZ
XVloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
�
Egradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulMatMulXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency2Generator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Ggradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1MatMul Generator/first_layer/leaky_reluXgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
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
gradients_1/AddN_8AddNfgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_85Generator/second_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_8:Generator/second_layer/batch_normalization/batchnorm/Rsqrt*
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
=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
7gradients_1/Generator/first_layer/leaky_relu_grad/zerosFill9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2=gradients_1/Generator/first_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualGreaterEqual$Generator/first_layer/leaky_relu/mul-Generator/first_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Ggradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/first_layer/leaky_relu_grad/Shape9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
5gradients_1/Generator/first_layer/leaky_relu_grad/SumSum8gradients_1/Generator/first_layer/leaky_relu_grad/SelectGgradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
9gradients_1/Generator/first_layer/leaky_relu_grad/ReshapeReshape5gradients_1/Generator/first_layer/leaky_relu_grad/Sum7gradients_1/Generator/first_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Igradients_1/Generator/first_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
gradients_1/AddN_9AddNLgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
�
Jgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Ogradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_9K^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Wgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_9P^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1
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
Fgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise_inWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	d�*
transpose_a(
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
VariableV2*
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape: 
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
beta2_power_1/initial_valueConst*
dtype0*
_output_shapes
: *,
_class"
 loc:@Generator/fake_image/bias*
valueB
 *w�?
�
beta2_power_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *,
_class"
 loc:@Generator/fake_image/bias
�
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
�
SGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   �   *
dtype0
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
8Generator/first_layer/fully_connected/kernel/Adam/AssignAssign1Generator/first_layer/fully_connected/kernel/AdamCGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0
�
6Generator/first_layer/fully_connected/kernel/Adam/readIdentity1Generator/first_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
UGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   �   *
dtype0
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
EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�*
T0
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
:Generator/first_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/first_layer/fully_connected/kernel/Adam_1EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	d�*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(
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
VariableV2*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0
�
6Generator/first_layer/fully_connected/bias/Adam/AssignAssign/Generator/first_layer/fully_connected/bias/AdamAGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(
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
6Generator/first_layer/fully_connected/bias/Adam_1/readIdentity1Generator/first_layer/fully_connected/bias/Adam_1*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:�*
T0
�
TGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"�      *
dtype0
�
JGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0
�
DGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillTGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorJGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
2Generator/second_layer/fully_connected/kernel/Adam
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
9Generator/second_layer/fully_connected/kernel/Adam/AssignAssign2Generator/second_layer/fully_connected/kernel/AdamDGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(
�
7Generator/second_layer/fully_connected/kernel/Adam/readIdentity2Generator/second_layer/fully_connected/kernel/Adam*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0
�
VGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
LGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0
�
FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillVGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0
�
4Generator/second_layer/fully_connected/kernel/Adam_1
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
9Generator/second_layer/fully_connected/kernel/Adam_1/readIdentity4Generator/second_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
�
BGenerator/second_layer/fully_connected/bias/Adam/Initializer/zerosConst*
_output_shapes	
:�*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB�*    *
dtype0
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
7Generator/second_layer/fully_connected/bias/Adam/AssignAssign0Generator/second_layer/fully_connected/bias/AdamBGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
5Generator/second_layer/fully_connected/bias/Adam/readIdentity0Generator/second_layer/fully_connected/bias/Adam*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:�
�
DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
valueB�*    *
dtype0
�
2Generator/second_layer/fully_connected/bias/Adam_1
VariableV2*
_output_shapes	
:�*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0
�
9Generator/second_layer/fully_connected/bias/Adam_1/AssignAssign2Generator/second_layer/fully_connected/bias/Adam_1DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(
�
7Generator/second_layer/fully_connected/bias/Adam_1/readIdentity2Generator/second_layer/fully_connected/bias/Adam_1*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:�
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
:Generator/second_layer/batch_normalization/gamma/Adam/readIdentity5Generator/second_layer/batch_normalization/gamma/Adam*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:�*
T0
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container 
�
>Generator/second_layer/batch_normalization/gamma/Adam_1/AssignAssign7Generator/second_layer/batch_normalization/gamma/Adam_1IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
�
;Generator/second_layer/batch_normalization/beta/Adam/AssignAssign4Generator/second_layer/batch_normalization/beta/AdamFGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
9Generator/second_layer/batch_normalization/beta/Adam/readIdentity4Generator/second_layer/batch_normalization/beta/Adam*
_output_shapes	
:�*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
�
HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB�*    *
dtype0
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
SGenerator/third_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0
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
UGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0
�
KGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0
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
_output_shapes
:
��*
shared_name *?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0
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
VariableV2*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
	container *
shape:�*
dtype0
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
8Generator/third_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/third_layer/fully_connected/bias/Adam_1CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(
�
6Generator/third_layer/fully_connected/bias/Adam_1/readIdentity1Generator/third_layer/fully_connected/bias/Adam_1*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:�*
T0
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
;Generator/third_layer/batch_normalization/gamma/Adam/AssignAssign4Generator/third_layer/batch_normalization/gamma/AdamFGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(
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
=Generator/third_layer/batch_normalization/gamma/Adam_1/AssignAssign6Generator/third_layer/batch_normalization/gamma/Adam_1HGenerator/third_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
;Generator/third_layer/batch_normalization/gamma/Adam_1/readIdentity6Generator/third_layer/batch_normalization/gamma/Adam_1*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:�*
T0
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
:Generator/third_layer/batch_normalization/beta/Adam/AssignAssign3Generator/third_layer/batch_normalization/beta/AdamEGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
8Generator/third_layer/batch_normalization/beta/Adam/readIdentity3Generator/third_layer/batch_normalization/beta/Adam*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:�
�
GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB�*    *
dtype0
�
5Generator/third_layer/batch_normalization/beta/Adam_1
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
<Generator/third_layer/batch_normalization/beta/Adam_1/AssignAssign5Generator/third_layer/batch_normalization/beta/Adam_1GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(
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
HGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
BGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zerosFillRGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
0Generator/last_layer/fully_connected/kernel/Adam
VariableV2* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0
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
TGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
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
9Generator/last_layer/fully_connected/kernel/Adam_1/AssignAssign2Generator/last_layer/fully_connected/kernel/Adam_1DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(
�
7Generator/last_layer/fully_connected/kernel/Adam_1/readIdentity2Generator/last_layer/fully_connected/kernel/Adam_1*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0
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
@Generator/last_layer/fully_connected/bias/Adam/Initializer/zerosFillPGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorFGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/Const*
_output_shapes	
:�*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0
�
.Generator/last_layer/fully_connected/bias/Adam
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
	container 
�
5Generator/last_layer/fully_connected/bias/Adam/AssignAssign.Generator/last_layer/fully_connected/bias/Adam@Generator/last_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
3Generator/last_layer/fully_connected/bias/Adam/readIdentity.Generator/last_layer/fully_connected/bias/Adam*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
_output_shapes	
:�*
T0
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
BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zerosFillRGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/Const*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:�*
T0
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
7Generator/last_layer/fully_connected/bias/Adam_1/AssignAssign0Generator/last_layer/fully_connected/bias/Adam_1BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
KGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0
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
:Generator/last_layer/batch_normalization/gamma/Adam/AssignAssign3Generator/last_layer/batch_normalization/gamma/AdamEGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
8Generator/last_layer/batch_normalization/gamma/Adam/readIdentity3Generator/last_layer/batch_normalization/gamma/Adam*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:�
�
WGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:�*
dtype0
�
MGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0
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
<Generator/last_layer/batch_normalization/gamma/Adam_1/AssignAssign5Generator/last_layer/batch_normalization/gamma/Adam_1GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
:Generator/last_layer/batch_normalization/gamma/Adam_1/readIdentity5Generator/last_layer/batch_normalization/gamma/Adam_1*
_output_shapes	
:�*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma
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
DGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zerosFillTGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorJGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:�
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
9Generator/last_layer/batch_normalization/beta/Adam/AssignAssign2Generator/last_layer/batch_normalization/beta/AdamDGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
7Generator/last_layer/batch_normalization/beta/Adam/readIdentity2Generator/last_layer/batch_normalization/beta/Adam*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:�
�
VGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:�*
dtype0
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
BGenerator/fake_image/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*.
_class$
" loc:@Generator/fake_image/kernel*
valueB"     *
dtype0
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
VariableV2*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
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
%Generator/fake_image/kernel/Adam/readIdentity Generator/fake_image/kernel/Adam*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��
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
4Generator/fake_image/kernel/Adam_1/Initializer/zerosFillDGenerator/fake_image/kernel/Adam_1/Initializer/zeros/shape_as_tensor:Generator/fake_image/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*.
_class$
" loc:@Generator/fake_image/kernel*

index_type0
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
)Generator/fake_image/kernel/Adam_1/AssignAssign"Generator/fake_image/kernel/Adam_14Generator/fake_image/kernel/Adam_1/Initializer/zeros*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
'Generator/fake_image/kernel/Adam_1/readIdentity"Generator/fake_image/kernel/Adam_1* 
_output_shapes
:
��*
T0*.
_class$
" loc:@Generator/fake_image/kernel
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container 
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
'Generator/fake_image/bias/Adam_1/AssignAssign Generator/fake_image/bias/Adam_12Generator/fake_image/bias/Adam_1/Initializer/zeros*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
Adam_1/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
Q
Adam_1/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
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
EAdam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdam	ApplyAdam-Generator/second_layer/fully_connected/kernel2Generator/second_layer/fully_connected/kernel/Adam4Generator/second_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
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
HAdam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam0Generator/second_layer/batch_normalization/gamma5Generator/second_layer/batch_normalization/gamma/Adam7Generator/second_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
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
BAdam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/third_layer/fully_connected/bias/Generator/third_layer/fully_connected/bias/Adam1Generator/third_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
GAdam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam/Generator/third_layer/batch_normalization/gamma4Generator/third_layer/batch_normalization/gamma/Adam6Generator/third_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma
�
FAdam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdam	ApplyAdam.Generator/third_layer/batch_normalization/beta3Generator/third_layer/batch_normalization/beta/Adam5Generator/third_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:�
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
FAdam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Generator/last_layer/batch_normalization/gamma3Generator/last_layer/batch_normalization/gamma/Adam5Generator/last_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
EAdam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Generator/last_layer/batch_normalization/beta2Generator/last_layer/batch_normalization/beta/Adam4Generator/last_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta
�
3Adam_1/update_Generator/fake_image/kernel/ApplyAdam	ApplyAdamGenerator/fake_image/kernel Generator/fake_image/kernel/Adam"Generator/fake_image/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*.
_class$
" loc:@Generator/fake_image/kernel
�
1Adam_1/update_Generator/fake_image/bias/ApplyAdam	ApplyAdamGenerator/fake_image/biasGenerator/fake_image/bias/Adam Generator/fake_image/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
use_nesterov( *
_output_shapes	
:�
�	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: *
T0
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: 
�	
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta22^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
_output_shapes
: *
N"k�F��l     ��C�	}�e����AJ��
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
u
Generator/noise_inPlaceholder*
shape:���������d*
dtype0*'
_output_shapes
:���������d
�
MGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
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
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *_&>*
dtype0
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
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/subSubKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/maxKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
: 
�
KGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulUGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub*
_output_shapes
:	d�*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel
�
GGenerator/first_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/first_layer/fully_connected/kernel/Initializer/random_uniform/min*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
,Generator/first_layer/fully_connected/kernel
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
<Generator/first_layer/fully_connected/bias/Initializer/zerosConst*
_output_shapes	
:�*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
valueB�*    *
dtype0
�
*Generator/first_layer/fully_connected/bias
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/first_layer/fully_connected/bias
�
1Generator/first_layer/fully_connected/bias/AssignAssign*Generator/first_layer/fully_connected/bias<Generator/first_layer/fully_connected/bias/Initializer/zeros*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
/Generator/first_layer/fully_connected/bias/readIdentity*Generator/first_layer/fully_connected/bias*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:�
�
,Generator/first_layer/fully_connected/MatMulMatMulGenerator/noise_in1Generator/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
-Generator/first_layer/fully_connected/BiasAddBiasAdd,Generator/first_layer/fully_connected/MatMul/Generator/first_layer/fully_connected/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
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
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/subSubLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
�
LGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulMulVGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformLGenerator/second_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel
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
VariableV2*
_output_shapes	
:�*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0
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
0Generator/second_layer/fully_connected/bias/readIdentity+Generator/second_layer/fully_connected/bias*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:�*
T0
�
-Generator/second_layer/fully_connected/MatMulMatMul Generator/first_layer/leaky_relu2Generator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
.Generator/second_layer/fully_connected/BiasAddBiasAdd-Generator/second_layer/fully_connected/MatMul0Generator/second_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
AGenerator/second_layer/batch_normalization/gamma/Initializer/onesConst*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
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
7Generator/second_layer/batch_normalization/gamma/AssignAssign0Generator/second_layer/batch_normalization/gammaAGenerator/second_layer/batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
5Generator/second_layer/batch_normalization/gamma/readIdentity0Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:�*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma
�
AGenerator/second_layer/batch_normalization/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB�*    
�
/Generator/second_layer/batch_normalization/beta
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
	container *
shape:�
�
=Generator/second_layer/batch_normalization/moving_mean/AssignAssign6Generator/second_layer/batch_normalization/moving_meanHGenerator/second_layer/batch_normalization/moving_mean/Initializer/zeros*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
;Generator/second_layer/batch_normalization/moving_mean/readIdentity6Generator/second_layer/batch_normalization/moving_mean*
T0*I
_class?
=;loc:@Generator/second_layer/batch_normalization/moving_mean*
_output_shapes	
:�
�
KGenerator/second_layer/batch_normalization/moving_variance/Initializer/onesConst*
_output_shapes	
:�*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
valueB�*  �?*
dtype0
�
:Generator/second_layer/batch_normalization/moving_variance
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
	container *
shape:�
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
?Generator/second_layer/batch_normalization/moving_variance/readIdentity:Generator/second_layer/batch_normalization/moving_variance*
T0*M
_classC
A?loc:@Generator/second_layer/batch_normalization/moving_variance*
_output_shapes	
:�

:Generator/second_layer/batch_normalization/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
8Generator/second_layer/batch_normalization/batchnorm/addAdd?Generator/second_layer/batch_normalization/moving_variance/read:Generator/second_layer/batch_normalization/batchnorm/add/y*
T0*
_output_shapes	
:�
�
:Generator/second_layer/batch_normalization/batchnorm/RsqrtRsqrt8Generator/second_layer/batch_normalization/batchnorm/add*
_output_shapes	
:�*
T0
�
8Generator/second_layer/batch_normalization/batchnorm/mulMul:Generator/second_layer/batch_normalization/batchnorm/Rsqrt5Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
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
8Generator/second_layer/batch_normalization/batchnorm/subSub4Generator/second_layer/batch_normalization/beta/read:Generator/second_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:�*
T0
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
UGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformMGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
��*

seed 
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
GGenerator/third_layer/fully_connected/kernel/Initializer/random_uniformAddKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/mulKGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform/min*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0
�
,Generator/third_layer/fully_connected/kernel
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
3Generator/third_layer/fully_connected/kernel/AssignAssign,Generator/third_layer/fully_connected/kernelGGenerator/third_layer/fully_connected/kernel/Initializer/random_uniform*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
1Generator/third_layer/fully_connected/kernel/readIdentity,Generator/third_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *=
_class3
1/loc:@Generator/third_layer/fully_connected/bias
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
/Generator/third_layer/fully_connected/bias/readIdentity*Generator/third_layer/fully_connected/bias*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:�*
T0
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
@Generator/third_layer/batch_normalization/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
valueB�*    
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
5Generator/third_layer/batch_normalization/beta/AssignAssign.Generator/third_layer/batch_normalization/beta@Generator/third_layer/batch_normalization/beta/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean
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
:Generator/third_layer/batch_normalization/moving_mean/readIdentity5Generator/third_layer/batch_normalization/moving_mean*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/third_layer/batch_normalization/moving_mean
�
JGenerator/third_layer/batch_normalization/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:�*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
valueB�*  �?
�
9Generator/third_layer/batch_normalization/moving_variance
VariableV2*
_output_shapes	
:�*
shared_name *L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
	container *
shape:�*
dtype0
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
>Generator/third_layer/batch_normalization/moving_variance/readIdentity9Generator/third_layer/batch_normalization/moving_variance*L
_classB
@>loc:@Generator/third_layer/batch_normalization/moving_variance*
_output_shapes	
:�*
T0
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
 Generator/third_layer/leaky_reluMaximum$Generator/third_layer/leaky_relu/mul9Generator/third_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
LGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      
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
TGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformLGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/shape*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
seed2 *
dtype0* 
_output_shapes
:
��*

seed 
�
JGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/subSubJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/maxJGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform/min*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
_output_shapes
: *
T0
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
2Generator/last_layer/fully_connected/kernel/AssignAssign+Generator/last_layer/fully_connected/kernelFGenerator/last_layer/fully_connected/kernel/Initializer/random_uniform*
T0*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
,Generator/last_layer/fully_connected/BiasAddBiasAdd+Generator/last_layer/fully_connected/MatMul.Generator/last_layer/fully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
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
3Generator/last_layer/batch_normalization/gamma/readIdentity.Generator/last_layer/batch_normalization/gamma*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
_output_shapes	
:�*
T0
�
OGenerator/last_layer/batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:�*
dtype0*
_output_shapes
:
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
2Generator/last_layer/batch_normalization/beta/readIdentity-Generator/last_layer/batch_normalization/beta*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:�
�
VGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB:�*
dtype0*
_output_shapes
:
�
LGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
FGenerator/last_layer/batch_normalization/moving_mean/Initializer/zerosFillVGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros/Const*
_output_shapes	
:�*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*

index_type0
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
;Generator/last_layer/batch_normalization/moving_mean/AssignAssign4Generator/last_layer/batch_normalization/moving_meanFGenerator/last_layer/batch_normalization/moving_mean/Initializer/zeros*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
9Generator/last_layer/batch_normalization/moving_mean/readIdentity4Generator/last_layer/batch_normalization/moving_mean*
T0*G
_class=
;9loc:@Generator/last_layer/batch_normalization/moving_mean*
_output_shapes	
:�
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
VariableV2*
shared_name *K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
?Generator/last_layer/batch_normalization/moving_variance/AssignAssign8Generator/last_layer/batch_normalization/moving_varianceIGenerator/last_layer/batch_normalization/moving_variance/Initializer/ones*
T0*K
_classA
?=loc:@Generator/last_layer/batch_normalization/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
8Generator/last_layer/batch_normalization/batchnorm/mul_2Mul9Generator/last_layer/batch_normalization/moving_mean/read6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
6Generator/last_layer/batch_normalization/batchnorm/subSub2Generator/last_layer/batch_normalization/beta/read8Generator/last_layer/batch_normalization/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
8Generator/last_layer/batch_normalization/batchnorm/add_1Add8Generator/last_layer/batch_normalization/batchnorm/mul_16Generator/last_layer/batch_normalization/batchnorm/sub*(
_output_shapes
:����������*
T0
j
%Generator/last_layer/leaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
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
:Generator/fake_image/kernel/Initializer/random_uniform/mulMulDGenerator/fake_image/kernel/Initializer/random_uniform/RandomUniform:Generator/fake_image/kernel/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��
�
6Generator/fake_image/kernel/Initializer/random_uniformAdd:Generator/fake_image/kernel/Initializer/random_uniform/mul:Generator/fake_image/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*.
_class$
" loc:@Generator/fake_image/kernel
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
+Generator/fake_image/bias/Initializer/zerosConst*,
_class"
 loc:@Generator/fake_image/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Generator/fake_image/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:�
�
 Generator/fake_image/bias/AssignAssignGenerator/fake_image/bias+Generator/fake_image/bias/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(*
_output_shapes	
:�
�
Generator/fake_image/bias/readIdentityGenerator/fake_image/bias*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:�
�
Generator/fake_image/MatMulMatMulGenerator/last_layer/leaky_relu Generator/fake_image/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
Generator/fake_image/BiasAddBiasAddGenerator/fake_image/MatMulGenerator/fake_image/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
r
Generator/fake_image/TanhTanhGenerator/fake_image/BiasAdd*
T0*(
_output_shapes
:����������
z
Discriminator/real_inPlaceholder*(
_output_shapes
:����������*
shape:����������*
dtype0
�
QDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB"     *
dtype0
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
ODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/mulMulYDiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/RandomUniformODiscriminator/first_layer/fully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel
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
@Discriminator/first_layer/fully_connected/bias/Initializer/zerosConst*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
.Discriminator/first_layer/fully_connected/bias
VariableV2*
_output_shapes	
:�*
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape:�*
dtype0
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
0Discriminator/first_layer/fully_connected/MatMulMatMulDiscriminator/real_in5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
1Discriminator/first_layer/fully_connected/BiasAddBiasAdd0Discriminator/first_layer/fully_connected/MatMul3Discriminator/first_layer/fully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
o
*Discriminator/first_layer/leaky_relu/alphaConst*
_output_shapes
: *
valueB
 *��L>*
dtype0
�
(Discriminator/first_layer/leaky_relu/mulMul*Discriminator/first_layer/leaky_relu/alpha1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
$Discriminator/first_layer/leaky_reluMaximum(Discriminator/first_layer/leaky_relu/mul1Discriminator/first_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
RDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB"      *
dtype0
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
PDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *��=*
dtype0
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
LDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniformAddPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/mulPDiscriminator/second_layer/fully_connected/kernel/Initializer/random_uniform/min*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0
�
1Discriminator/second_layer/fully_connected/kernel
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
ADiscriminator/second_layer/fully_connected/bias/Initializer/zerosConst*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
/Discriminator/second_layer/fully_connected/bias
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
6Discriminator/second_layer/fully_connected/bias/AssignAssign/Discriminator/second_layer/fully_connected/biasADiscriminator/second_layer/fully_connected/bias/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
4Discriminator/second_layer/fully_connected/bias/readIdentity/Discriminator/second_layer/fully_connected/bias*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:�
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
+Discriminator/second_layer/leaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
�
)Discriminator/second_layer/leaky_relu/mulMul+Discriminator/second_layer/leaky_relu/alpha2Discriminator/second_layer/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
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
4Discriminator/prob/kernel/Initializer/random_uniformAdd8Discriminator/prob/kernel/Initializer/random_uniform/mul8Discriminator/prob/kernel/Initializer/random_uniform/min*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�*
T0
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
 Discriminator/prob/kernel/AssignAssignDiscriminator/prob/kernel4Discriminator/prob/kernel/Initializer/random_uniform*
_output_shapes
:	�*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(
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
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container 
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
+Discriminator/second_layer_1/leaky_relu/mulMul-Discriminator/second_layer_1/leaky_relu/alpha4Discriminator/second_layer_1/fully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
'Discriminator/second_layer_1/leaky_reluMaximum+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
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
	ones_likeFillones_like/Shapeones_like/Const*

index_type0*'
_output_shapes
:���������*
T0
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
logistic_loss/mulMulDiscriminator/prob/BiasAdd	ones_like*'
_output_shapes
:���������*
T0
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
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*'
_output_shapes
:���������
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
`
MeanMeanlogistic_lossConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
g

zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:���������*
T0
w
logistic_loss_1/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss_1/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_1/zeros_like*'
_output_shapes
:���������*
T0
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
v
logistic_loss_1/mulMulDiscriminator/prob_1/BiasAdd
zeros_like*
T0*'
_output_shapes
:���������
y
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*'
_output_shapes
:���������*
T0
f
logistic_loss_1/ExpExplogistic_loss_1/Select_1*
T0*'
_output_shapes
:���������
e
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*
T0*'
_output_shapes
:���������
t
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*'
_output_shapes
:���������*
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
ones_like_1/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
}
ones_like_1Fillones_like_1/Shapeones_like_1/Const*'
_output_shapes
:���������*
T0*

index_type0
w
logistic_loss_2/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:���������*
T0
�
logistic_loss_2/GreaterEqualGreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:���������
�
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualDiscriminator/prob_1/BiasAddlogistic_loss_2/zeros_like*
T0*'
_output_shapes
:���������
j
logistic_loss_2/NegNegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
�
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/NegDiscriminator/prob_1/BiasAdd*
T0*'
_output_shapes
:���������
w
logistic_loss_2/mulMulDiscriminator/prob_1/BiasAddones_like_1*
T0*'
_output_shapes
:���������
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
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*'
_output_shapes
:���������*
T0
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
+gradients/add_grad/tuple/control_dependencyIdentitygradients/Fill$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
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
f
gradients/Mean_grad/ShapeShapelogistic_loss*
T0*
out_type0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
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
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:���������*
T0
t
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
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
�
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
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
gradients/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
g
gradients/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
a
gradients/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
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
s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0*
_output_shapes
:
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
_output_shapes
:*
T0*
out_type0
�
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1
�
5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape*'
_output_shapes
:���������
�
7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1*'
_output_shapes
:���������
w
$gradients/logistic_loss_1_grad/ShapeShapelogistic_loss_1/sub*
_output_shapes
:*
T0*
out_type0
{
&gradients/logistic_loss_1_grad/Shape_1Shapelogistic_loss_1/Log1p*
_output_shapes
:*
T0*
out_type0
�
4gradients/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/logistic_loss_1_grad/Shape&gradients/logistic_loss_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
"gradients/logistic_loss_1_grad/SumSumgradients/Mean_1_grad/truediv4gradients/logistic_loss_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
&gradients/logistic_loss_1_grad/ReshapeReshape"gradients/logistic_loss_1_grad/Sum$gradients/logistic_loss_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
$gradients/logistic_loss_1_grad/Sum_1Sumgradients/Mean_1_grad/truediv6gradients/logistic_loss_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss_1_grad/Reshape_1Reshape$gradients/logistic_loss_1_grad/Sum_1&gradients/logistic_loss_1_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
/gradients/logistic_loss_1_grad/tuple/group_depsNoOp'^gradients/logistic_loss_1_grad/Reshape)^gradients/logistic_loss_1_grad/Reshape_1
�
7gradients/logistic_loss_1_grad/tuple/control_dependencyIdentity&gradients/logistic_loss_1_grad/Reshape0^gradients/logistic_loss_1_grad/tuple/group_deps*9
_class/
-+loc:@gradients/logistic_loss_1_grad/Reshape*'
_output_shapes
:���������*
T0
�
9gradients/logistic_loss_1_grad/tuple/control_dependency_1Identity(gradients/logistic_loss_1_grad/Reshape_10^gradients/logistic_loss_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@gradients/logistic_loss_1_grad/Reshape_1
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
�
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
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
�
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1
�
9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape*'
_output_shapes
:���������
�
;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1*'
_output_shapes
:���������
�
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*'
_output_shapes
:���������*
T0
�
-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*'
_output_shapes
:���������*
T0
�
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:���������*
T0
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
*gradients/logistic_loss_1/sub_grad/ReshapeReshape&gradients/logistic_loss_1/sub_grad/Sum(gradients/logistic_loss_1/sub_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
(gradients/logistic_loss_1/sub_grad/Sum_1Sum7gradients/logistic_loss_1_grad/tuple/control_dependency:gradients/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
z
&gradients/logistic_loss_1/sub_grad/NegNeg(gradients/logistic_loss_1/sub_grad/Sum_1*
T0*
_output_shapes
:
�
,gradients/logistic_loss_1/sub_grad/Reshape_1Reshape&gradients/logistic_loss_1/sub_grad/Neg*gradients/logistic_loss_1/sub_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
3gradients/logistic_loss_1/sub_grad/tuple/group_depsNoOp+^gradients/logistic_loss_1/sub_grad/Reshape-^gradients/logistic_loss_1/sub_grad/Reshape_1
�
;gradients/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity*gradients/logistic_loss_1/sub_grad/Reshape4^gradients/logistic_loss_1/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@gradients/logistic_loss_1/sub_grad/Reshape
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
Reciprocal(gradients/logistic_loss_1/Log1p_grad/add*'
_output_shapes
:���������*
T0
�
(gradients/logistic_loss_1/Log1p_grad/mulMul9gradients/logistic_loss_1_grad/tuple/control_dependency_1/gradients/logistic_loss_1/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeDiscriminator/prob/BiasAdd*
T0*'
_output_shapes
:���������
�
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:���������
�
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1
�
<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
�
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1*'
_output_shapes
:���������
�
&gradients/logistic_loss/mul_grad/ShapeShapeDiscriminator/prob/BiasAdd*
out_type0*
_output_shapes
:*
T0
q
(gradients/logistic_loss/mul_grad/Shape_1Shape	ones_like*
T0*
out_type0*
_output_shapes
:
�
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*
T0*'
_output_shapes
:���������
�
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&gradients/logistic_loss/mul_grad/Mul_1MulDiscriminator/prob/BiasAdd;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1
�
9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape*'
_output_shapes
:���������
�
;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
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
.gradients/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual0gradients/logistic_loss_1/Select_grad/zeros_like;gradients/logistic_loss_1/sub_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
6gradients/logistic_loss_1/Select_grad/tuple/group_depsNoOp-^gradients/logistic_loss_1/Select_grad/Select/^gradients/logistic_loss_1/Select_grad/Select_1
�
>gradients/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity,gradients/logistic_loss_1/Select_grad/Select7^gradients/logistic_loss_1/Select_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select
�
@gradients/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity.gradients/logistic_loss_1/Select_grad/Select_17^gradients/logistic_loss_1/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_grad/Select_1*'
_output_shapes
:���������
�
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
�
8gradients/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/logistic_loss_1/mul_grad/Shape*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
&gradients/logistic_loss_1/mul_grad/MulMul=gradients/logistic_loss_1/sub_grad/tuple/control_dependency_1
zeros_like*'
_output_shapes
:���������*
T0
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
(gradients/logistic_loss_1/mul_grad/Sum_1Sum(gradients/logistic_loss_1/mul_grad/Mul_1:gradients/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
,gradients/logistic_loss_1/mul_grad/Reshape_1Reshape(gradients/logistic_loss_1/mul_grad/Sum_1*gradients/logistic_loss_1/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:���������
�
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*'
_output_shapes
:���������*
T0
�
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:���������
�
6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
�
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select*'
_output_shapes
:���������*
T0
�
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1
�
2gradients/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*'
_output_shapes
:���������*
T0
�
.gradients/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual&gradients/logistic_loss_1/Exp_grad/mul2gradients/logistic_loss_1/Select_1_grad/zeros_like*
T0*'
_output_shapes
:���������
�
0gradients/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients/logistic_loss_1/Select_1_grad/zeros_like&gradients/logistic_loss_1/Exp_grad/mul*'
_output_shapes
:���������*
T0
�
8gradients/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp/^gradients/logistic_loss_1/Select_1_grad/Select1^gradients/logistic_loss_1/Select_1_grad/Select_1
�
@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity.gradients/logistic_loss_1/Select_1_grad/Select9^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/logistic_loss_1/Select_1_grad/Select*'
_output_shapes
:���������
�
Bgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity0gradients/logistic_loss_1/Select_1_grad/Select_19^gradients/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/logistic_loss_1/Select_1_grad/Select_1*'
_output_shapes
:���������
�
$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
&gradients/logistic_loss_1/Neg_grad/NegNeg@gradients/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*
N*'
_output_shapes
:���������
�
5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:
�
:gradients/Discriminator/prob/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN6^gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad
�
Bgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select*'
_output_shapes
:���������
�
Dgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad;^gradients/Discriminator/prob/BiasAdd_grad/tuple/group_deps*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
gradients/AddN_1AddN>gradients/logistic_loss_1/Select_grad/tuple/control_dependency;gradients/logistic_loss_1/mul_grad/tuple/control_dependencyBgradients/logistic_loss_1/Select_1_grad/tuple/control_dependency_1&gradients/logistic_loss_1/Neg_grad/Neg*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select*
N*'
_output_shapes
:���������*
T0
�
7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
T0*
data_formatNHWC*
_output_shapes
:
�
<gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_18^gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
�
Dgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_1=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients/logistic_loss_1/Select_grad/Select
�
Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad=^gradients/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*J
_class@
><loc:@gradients/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
�
/gradients/Discriminator/prob/MatMul_grad/MatMulMatMulBgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
Cgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Identity1gradients/Discriminator/prob/MatMul_grad/MatMul_1:^gradients/Discriminator/prob/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
1gradients/Discriminator/prob_1/MatMul_grad/MatMulMatMulDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
3gradients/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluDgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	�*
transpose_a(
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
gradients/AddN_2AddNDgradients/Discriminator/prob/BiasAdd_grad/tuple/control_dependency_1Fgradients/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
:*
T0*H
_class>
<:loc:@gradients/Discriminator/prob/BiasAdd_grad/BiasAddGrad
�
:gradients/Discriminator/second_layer/leaky_relu_grad/ShapeShape)Discriminator/second_layer/leaky_relu/mul*
out_type0*
_output_shapes
:*
T0
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
@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
:gradients/Discriminator/second_layer/leaky_relu_grad/zerosFill<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_2@gradients/Discriminator/second_layer/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:����������*
T0
�
Agradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual)Discriminator/second_layer/leaky_relu/mul2Discriminator/second_layer/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
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
=gradients/Discriminator/second_layer/leaky_relu_grad/Select_1SelectAgradients/Discriminator/second_layer/leaky_relu_grad/GreaterEqual:gradients/Discriminator/second_layer/leaky_relu_grad/zerosAgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
>gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1Reshape:gradients/Discriminator/second_layer/leaky_relu_grad/Sum_1<gradients/Discriminator/second_layer/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
<gradients/Discriminator/second_layer_1/leaky_relu_grad/ShapeShape+Discriminator/second_layer_1/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
>gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
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
?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1SelectCgradients/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqual<gradients/Discriminator/second_layer_1/leaky_relu_grad/zerosCgradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
:gradients/Discriminator/second_layer_1/leaky_relu_grad/SumSum=gradients/Discriminator/second_layer_1/leaky_relu_grad/SelectLgradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
>gradients/Discriminator/second_layer_1/leaky_relu_grad/ReshapeReshape:gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum<gradients/Discriminator/second_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
<gradients/Discriminator/second_layer_1/leaky_relu_grad/Sum_1Sum?gradients/Discriminator/second_layer_1/leaky_relu_grad/Select_1Ngradients/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Qgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1H^gradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_3AddNCgradients/Discriminator/prob/MatMul_grad/tuple/control_dependency_1Egradients/Discriminator/prob_1/MatMul_grad/tuple/control_dependency_1*
T0*D
_class:
86loc:@gradients/Discriminator/prob/MatMul_grad/MatMul_1*
N*
_output_shapes
:	�
�
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Mul+Discriminator/second_layer/leaky_relu/alphaMgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1Sum>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Mul_1Pgradients/Discriminator/second_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Bgradients/Discriminator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape>gradients/Discriminator/second_layer/leaky_relu/mul_grad/Sum_1@gradients/Discriminator/second_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/SumSum>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/MulPgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Bgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeReshape>gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityDgradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1L^gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
�
gradients/AddN_4AddNOgradients/Discriminator/second_layer/leaky_relu_grad/tuple/control_dependency_1Sgradients/Discriminator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@gradients/Discriminator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Mgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Rgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4N^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Zgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4S^gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
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
gradients/AddN_5AddNQgradients/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Ugradients/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ogradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_5*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Tgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_5P^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
\gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_5U^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*S
_classI
GEloc:@gradients/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityOgradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradU^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Ggradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMulMatMulZgradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependencyIdentityIgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulT^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps*\
_classR
PNloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityKgradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1T^gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*^
_classT
RPloc:@gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1
�
gradients/AddN_6AddN\gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1^gradients/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*`
_classV
TRloc:@gradients/Discriminator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
9gradients/Discriminator/first_layer/leaky_relu_grad/ShapeShape(Discriminator/first_layer/leaky_relu/mul*
_output_shapes
:*
T0*
out_type0
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
Igradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/Discriminator/first_layer/leaky_relu_grad/Shape;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/Discriminator/first_layer/leaky_relu_grad/SelectSelect@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqualYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency9gradients/Discriminator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
<gradients/Discriminator/first_layer/leaky_relu_grad/Select_1Select@gradients/Discriminator/first_layer/leaky_relu_grad/GreaterEqual9gradients/Discriminator/first_layer/leaky_relu_grad/zerosYgradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
7gradients/Discriminator/first_layer/leaky_relu_grad/SumSum:gradients/Discriminator/first_layer/leaky_relu_grad/SelectIgradients/Discriminator/first_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1Reshape9gradients/Discriminator/first_layer/leaky_relu_grad/Sum_1;gradients/Discriminator/first_layer/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
Ngradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Identity=gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1E^gradients/Discriminator/first_layer/leaky_relu_grad/tuple/group_deps*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
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
=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape[gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
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
9gradients/Discriminator/first_layer_1/leaky_relu_grad/SumSum<gradients/Discriminator/first_layer_1/leaky_relu_grad/SelectKgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
=gradients/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape9gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum;gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum>gradients/Discriminator/first_layer_1/leaky_relu_grad/Select_1Mgradients/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1Reshape;gradients/Discriminator/first_layer_1/leaky_relu_grad/Sum_1=gradients/Discriminator/first_layer_1/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
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
Pgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Identity?gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1G^gradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/group_deps*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
gradients/AddN_7AddN[gradients/Discriminator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1]gradients/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1*
N* 
_output_shapes
:
��*
T0*\
_classR
PNloc:@gradients/Discriminator/second_layer/fully_connected/MatMul_grad/MatMul_1
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
?gradients/Discriminator/first_layer/leaky_relu/mul_grad/ReshapeReshape;gradients/Discriminator/first_layer/leaky_relu/mul_grad/Sum=gradients/Discriminator/first_layer/leaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
Cgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1Reshape?gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1Agradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Jgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpB^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeD^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1
�
Rgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityAgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/ReshapeK^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*T
_classJ
HFloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape
�
Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityCgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1K^gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/group_deps*V
_classL
JHloc:@gradients/Discriminator/first_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
gradients/AddN_8AddNNgradients/Discriminator/first_layer/leaky_relu_grad/tuple/control_dependency_1Rgradients/Discriminator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*
N
�
Lgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Qgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8M^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Ygradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8R^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/Discriminator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradR^gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
gradients/AddN_9AddNPgradients/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Tgradients/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
Ngradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Sgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_9O^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_9T^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*R
_classH
FDloc:@gradients/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradT^gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Fgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMulMatMulYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Hgradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/real_inYgradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
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
Hgradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul[gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
gradients/AddN_10AddN[gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1]gradients/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0*_
_classU
SQloc:@gradients/Discriminator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
N
�
gradients/AddN_11AddNZgradients/Discriminator/first_layer/fully_connected/MatMul_grad/tuple/control_dependency_1\gradients/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
T0*[
_classQ
OMloc:@gradients/Discriminator/first_layer/fully_connected/MatMul_grad/MatMul_1*
N
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
VariableV2*
_output_shapes
: *
shared_name *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
	container *
shape: *
dtype0
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
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
valueB
 *w�?
�
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
beta2_power/readIdentitybeta2_power*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: *
T0
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
GDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zerosFillWDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorMDiscriminator/first_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
5Discriminator/first_layer/fully_connected/kernel/Adam
VariableV2* 
_output_shapes
:
��*
shared_name *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0
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
ODiscriminator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
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
VariableV2*C
_class9
75loc:@Discriminator/first_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
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
:Discriminator/first_layer/fully_connected/bias/Adam/AssignAssign3Discriminator/first_layer/fully_connected/bias/AdamEDiscriminator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias
�
8Discriminator/first_layer/fully_connected/bias/Adam/readIdentity3Discriminator/first_layer/fully_connected/bias/Adam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes	
:�
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
NDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
valueB
 *    *
dtype0
�
HDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillXDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorNDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
6Discriminator/second_layer/fully_connected/kernel/Adam
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
=Discriminator/second_layer/fully_connected/kernel/Adam/AssignAssign6Discriminator/second_layer/fully_connected/kernel/AdamHDiscriminator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
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
=Discriminator/second_layer/fully_connected/kernel/Adam_1/readIdentity8Discriminator/second_layer/fully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel
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
;Discriminator/second_layer/fully_connected/bias/Adam/AssignAssign4Discriminator/second_layer/fully_connected/bias/AdamFDiscriminator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias
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
=Discriminator/second_layer/fully_connected/bias/Adam_1/AssignAssign6Discriminator/second_layer/fully_connected/bias/Adam_1HDiscriminator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
;Discriminator/second_layer/fully_connected/bias/Adam_1/readIdentity6Discriminator/second_layer/fully_connected/bias/Adam_1*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
_output_shapes	
:�*
T0
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
%Discriminator/prob/kernel/Adam/AssignAssignDiscriminator/prob/kernel/Adam0Discriminator/prob/kernel/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�
�
#Discriminator/prob/kernel/Adam/readIdentityDiscriminator/prob/kernel/Adam*,
_class"
 loc:@Discriminator/prob/kernel*
_output_shapes
:	�*
T0
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
'Discriminator/prob/kernel/Adam_1/AssignAssign Discriminator/prob/kernel/Adam_12Discriminator/prob/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@Discriminator/prob/kernel*
validate_shape(*
_output_shapes
:	�
�
%Discriminator/prob/kernel/Adam_1/readIdentity Discriminator/prob/kernel/Adam_1*
_output_shapes
:	�*
T0*,
_class"
 loc:@Discriminator/prob/kernel
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
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name **
_class 
loc:@Discriminator/prob/bias*
	container 
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
%Discriminator/prob/bias/Adam_1/AssignAssignDiscriminator/prob/bias/Adam_10Discriminator/prob/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@Discriminator/prob/bias*
validate_shape(
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
Adam/beta2Adam/epsilongradients/AddN_11*
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
Adam/beta2Adam/epsilongradients/AddN_10*
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
Adam/beta2Adam/epsilongradients/AddN_7*D
_class:
86loc:@Discriminator/second_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0
�
EAdam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdam	ApplyAdam/Discriminator/second_layer/fully_connected/bias4Discriminator/second_layer/fully_connected/bias/Adam6Discriminator/second_layer/fully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*
T0*B
_class8
64loc:@Discriminator/second_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
/Adam/update_Discriminator/prob/kernel/ApplyAdam	ApplyAdamDiscriminator/prob/kernelDiscriminator/prob/kernel/Adam Discriminator/prob/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
_output_shapes
:	�*
use_locking( *
T0*,
_class"
 loc:@Discriminator/prob/kernel*
use_nesterov( 
�
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
Adam/beta2E^Adam/update_Discriminator/first_layer/fully_connected/bias/ApplyAdamG^Adam/update_Discriminator/first_layer/fully_connected/kernel/ApplyAdam.^Adam/update_Discriminator/prob/bias/ApplyAdam0^Adam/update_Discriminator/prob/kernel/ApplyAdamF^Adam/update_Discriminator/second_layer/fully_connected/bias/ApplyAdamH^Adam/update_Discriminator/second_layer/fully_connected/kernel/ApplyAdam*
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *
use_locking( *
T0*A
_class7
53loc:@Discriminator/first_layer/fully_connected/bias*
validate_shape(
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
%gradients_1/Mean_2_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
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
�
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
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
gradients_1/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_1/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
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
�
gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0
�
 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
�
gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*'
_output_shapes
:���������*
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
�
6gradients_1/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_2_grad/Shape(gradients_1/logistic_loss_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$gradients_1/logistic_loss_2_grad/SumSumgradients_1/Mean_2_grad/truediv6gradients_1/logistic_loss_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(gradients_1/logistic_loss_2_grad/ReshapeReshape$gradients_1/logistic_loss_2_grad/Sum&gradients_1/logistic_loss_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&gradients_1/logistic_loss_2_grad/Sum_1Sumgradients_1/Mean_2_grad/truediv8gradients_1/logistic_loss_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*gradients_1/logistic_loss_2_grad/Reshape_1Reshape&gradients_1/logistic_loss_2_grad/Sum_1(gradients_1/logistic_loss_2_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
1gradients_1/logistic_loss_2_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_2_grad/Reshape+^gradients_1/logistic_loss_2_grad/Reshape_1
�
9gradients_1/logistic_loss_2_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_2_grad/Reshape2^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_2_grad/Reshape*'
_output_shapes
:���������
�
;gradients_1/logistic_loss_2_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_2_grad/Reshape_12^gradients_1/logistic_loss_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss_2_grad/Reshape_1*'
_output_shapes
:���������
�
*gradients_1/logistic_loss_2/sub_grad/ShapeShapelogistic_loss_2/Select*
_output_shapes
:*
T0*
out_type0

,gradients_1/logistic_loss_2/sub_grad/Shape_1Shapelogistic_loss_2/mul*
T0*
out_type0*
_output_shapes
:
�
:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/sub_grad/Shape,gradients_1/logistic_loss_2/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
(gradients_1/logistic_loss_2/sub_grad/SumSum9gradients_1/logistic_loss_2_grad/tuple/control_dependency:gradients_1/logistic_loss_2/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
,gradients_1/logistic_loss_2/sub_grad/ReshapeReshape(gradients_1/logistic_loss_2/sub_grad/Sum*gradients_1/logistic_loss_2/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
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
�
.gradients_1/logistic_loss_2/sub_grad/Reshape_1Reshape(gradients_1/logistic_loss_2/sub_grad/Neg,gradients_1/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
5gradients_1/logistic_loss_2/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/sub_grad/Reshape/^gradients_1/logistic_loss_2/sub_grad/Reshape_1
�
=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/sub_grad/Reshape6^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/sub_grad/Reshape
�
?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/sub_grad/Reshape_16^gradients_1/logistic_loss_2/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/sub_grad/Reshape_1*'
_output_shapes
:���������
�
,gradients_1/logistic_loss_2/Log1p_grad/add/xConst<^gradients_1/logistic_loss_2_grad/tuple/control_dependency_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
*gradients_1/logistic_loss_2/Log1p_grad/addAdd,gradients_1/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*'
_output_shapes
:���������*
T0
�
1gradients_1/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_2/Log1p_grad/add*
T0*'
_output_shapes
:���������
�
*gradients_1/logistic_loss_2/Log1p_grad/mulMul;gradients_1/logistic_loss_2_grad/tuple/control_dependency_11gradients_1/logistic_loss_2/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:���������
�
2gradients_1/logistic_loss_2/Select_grad/zeros_like	ZerosLikeDiscriminator/prob_1/BiasAdd*'
_output_shapes
:���������*
T0
�
.gradients_1/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_2/Select_grad/zeros_like*'
_output_shapes
:���������*
T0
�
0gradients_1/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients_1/logistic_loss_2/Select_grad/zeros_like=gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
�
8gradients_1/logistic_loss_2/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_2/Select_grad/Select1^gradients_1/logistic_loss_2/Select_grad/Select_1
�
@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_2/Select_grad/Select9^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:���������
�
Bgradients_1/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_2/Select_grad/Select_19^gradients_1/logistic_loss_2/Select_grad/tuple/group_deps*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_grad/Select_1*'
_output_shapes
:���������*
T0
�
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
�
:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_2/mul_grad/Shape,gradients_1/logistic_loss_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
(gradients_1/logistic_loss_2/mul_grad/MulMul?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1ones_like_1*
T0*'
_output_shapes
:���������
�
(gradients_1/logistic_loss_2/mul_grad/SumSum(gradients_1/logistic_loss_2/mul_grad/Mul:gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
,gradients_1/logistic_loss_2/mul_grad/ReshapeReshape(gradients_1/logistic_loss_2/mul_grad/Sum*gradients_1/logistic_loss_2/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
*gradients_1/logistic_loss_2/mul_grad/Mul_1MulDiscriminator/prob_1/BiasAdd?gradients_1/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
*gradients_1/logistic_loss_2/mul_grad/Sum_1Sum*gradients_1/logistic_loss_2/mul_grad/Mul_1<gradients_1/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
.gradients_1/logistic_loss_2/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_2/mul_grad/Sum_1,gradients_1/logistic_loss_2/mul_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
5gradients_1/logistic_loss_2/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_2/mul_grad/Reshape/^gradients_1/logistic_loss_2/mul_grad/Reshape_1
�
=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_2/mul_grad/Reshape6^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients_1/logistic_loss_2/mul_grad/Reshape
�
?gradients_1/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_2/mul_grad/Reshape_16^gradients_1/logistic_loss_2/mul_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/logistic_loss_2/mul_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
(gradients_1/logistic_loss_2/Exp_grad/mulMul*gradients_1/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*'
_output_shapes
:���������*
T0
�
4gradients_1/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*'
_output_shapes
:���������*
T0
�
0gradients_1/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual(gradients_1/logistic_loss_2/Exp_grad/mul4gradients_1/logistic_loss_2/Select_1_grad/zeros_like*
T0*'
_output_shapes
:���������
�
2gradients_1/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual4gradients_1/logistic_loss_2/Select_1_grad/zeros_like(gradients_1/logistic_loss_2/Exp_grad/mul*'
_output_shapes
:���������*
T0
�
:gradients_1/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_2/Select_1_grad/Select3^gradients_1/logistic_loss_2/Select_1_grad/Select_1
�
Bgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_2/Select_1_grad/Select;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_2/Select_1_grad/Select*'
_output_shapes
:���������
�
Dgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_2/Select_1_grad/Select_1;^gradients_1/logistic_loss_2/Select_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*E
_class;
97loc:@gradients_1/logistic_loss_2/Select_1_grad/Select_1
�
(gradients_1/logistic_loss_2/Neg_grad/NegNegBgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
gradients_1/AddNAddN@gradients_1/logistic_loss_2/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_2/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_2/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_2/Neg_grad/Neg*
N*'
_output_shapes
:���������*
T0*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select
�
9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes
:
�
>gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN:^gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
�
Fgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/logistic_loss_2/Select_grad/Select*'
_output_shapes
:���������*
T0
�
Hgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad?^gradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*L
_classB
@>loc:@gradients_1/Discriminator/prob_1/BiasAdd_grad/BiasAddGrad
�
3gradients_1/Discriminator/prob_1/MatMul_grad/MatMulMatMulFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/prob/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
5gradients_1/Discriminator/prob_1/MatMul_grad/MatMul_1MatMul'Discriminator/second_layer_1/leaky_reluFgradients_1/Discriminator/prob_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
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
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1Shape4Discriminator/second_layer_1/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2ShapeEgradients_1/Discriminator/prob_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/zerosFill@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_2Dgradients_1/Discriminator/second_layer_1/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Egradients_1/Discriminator/second_layer_1/leaky_relu_grad/GreaterEqualGreaterEqual+Discriminator/second_layer_1/leaky_relu/mul4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Ngradients_1/Discriminator/second_layer_1/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Sgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1IdentityBgradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1J^gradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
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
@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/MulMulQgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency4Discriminator/second_layer_1/fully_connected/BiasAdd*
T0*(
_output_shapes
:����������
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
Bgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1SumBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Mul_1Tgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Fgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1ReshapeBgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Sum_1Dgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Mgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_depsNoOpE^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeG^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1
�
Ugradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependencyIdentityDgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/ReshapeN^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*W
_classM
KIloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape
�
Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1IdentityFgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1N^gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
gradients_1/AddN_1AddNSgradients_1/Discriminator/second_layer_1/leaky_relu_grad/tuple/control_dependency_1Wgradients_1/Discriminator/second_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Qgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Vgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1R^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1W^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*U
_classK
IGloc:@gradients_1/Discriminator/second_layer_1/leaky_relu_grad/Reshape_1
�
`gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityQgradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGradW^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*d
_classZ
XVloc:@gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
Kgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMulMatMul^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency6Discriminator/second_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Mgradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/MatMul_1MatMul&Discriminator/first_layer_1/leaky_relu^gradients_1/Discriminator/second_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
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
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape_2Shape]gradients_1/Discriminator/second_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
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
;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SumSum>gradients_1/Discriminator/first_layer_1/leaky_relu_grad/SelectMgradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
?gradients_1/Discriminator/first_layer_1/leaky_relu_grad/ReshapeReshape;gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
=gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Sum_1Sum@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Select_1Ogradients_1/Discriminator/first_layer_1/leaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/SumSum?gradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/MulQgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Agradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Sum_1SumAgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/Mul_1Sgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
gradients_1/AddN_2AddNRgradients_1/Discriminator/first_layer_1/leaky_relu_grad/tuple/control_dependency_1Vgradients_1/Discriminator/first_layer_1/leaky_relu/mul_grad/tuple/control_dependency_1*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������*
T0
�
Pgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ugradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_2Q^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad
�
]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_2V^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*T
_classJ
HFloc:@gradients_1/Discriminator/first_layer_1/leaky_relu_grad/Reshape_1
�
_gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGradV^gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Jgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMulMatMul]gradients_1/Discriminator/first_layer_1/fully_connected/BiasAdd_grad/tuple/control_dependency5Discriminator/first_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityLgradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1U^gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/group_deps*_
_classU
SQloc:@gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
3gradients_1/Generator/fake_image/Tanh_grad/TanhGradTanhGradGenerator/fake_image/Tanh\gradients_1/Discriminator/first_layer_1/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
9gradients_1/Generator/fake_image/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients_1/Generator/fake_image/Tanh_grad/TanhGrad*
_output_shapes	
:�*
T0*
data_formatNHWC
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
6gradients_1/Generator/last_layer/leaky_relu_grad/zerosFill8gradients_1/Generator/last_layer/leaky_relu_grad/Shape_2<gradients_1/Generator/last_layer/leaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
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
9gradients_1/Generator/last_layer/leaky_relu_grad/Select_1Select=gradients_1/Generator/last_layer/leaky_relu_grad/GreaterEqual6gradients_1/Generator/last_layer/leaky_relu_grad/zerosEgradients_1/Generator/fake_image/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
4gradients_1/Generator/last_layer/leaky_relu_grad/SumSum7gradients_1/Generator/last_layer/leaky_relu_grad/SelectFgradients_1/Generator/last_layer/leaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Igradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependencyIdentity8gradients_1/Generator/last_layer/leaky_relu_grad/ReshapeB^gradients_1/Generator/last_layer/leaky_relu_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������
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
<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1Shape8Generator/last_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
Jgradients_1/Generator/last_layer/leaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8gradients_1/Generator/last_layer/leaky_relu/mul_grad/MulMulIgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency8Generator/last_layer/batch_normalization/batchnorm/add_1*
T0*(
_output_shapes
:����������
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
>gradients_1/Generator/last_layer/leaky_relu/mul_grad/Reshape_1Reshape:gradients_1/Generator/last_layer/leaky_relu/mul_grad/Sum_1<gradients_1/Generator/last_layer/leaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
gradients_1/AddN_3AddNKgradients_1/Generator/last_layer/leaky_relu_grad/tuple/control_dependency_1Ogradients_1/Generator/last_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*M
_classC
A?loc:@gradients_1/Generator/last_layer/leaky_relu_grad/Reshape_1
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape8Generator/last_layer/batch_normalization/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_3_gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeMgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_3agradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
Tshape0*
_output_shapes	
:�*
T0
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
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeShape,Generator/last_layer/fully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/ShapeQgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/MulMulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency6Generator/last_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/SumSumMgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1agradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Sgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Qgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
Tshape0*
_output_shapes	
:�*
T0
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
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentitydgradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1Y^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
bgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/NegY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/Neg*
_output_shapes	
:�*
T0
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
Ogradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulbgradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_19Generator/last_layer/batch_normalization/moving_mean/read*
_output_shapes	
:�*
T0
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
dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1[^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Cgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulMatMulVgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency0Generator/last_layer/fully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
Ugradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityCgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMulN^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps*V
_classL
JHloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
Wgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityEgradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1N^gradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*X
_classN
LJloc:@gradients_1/Generator/last_layer/fully_connected/MatMul_grad/MatMul_1
�
gradients_1/AddN_4AddNdgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1dgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*f
_class\
ZXloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Kgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_43Generator/last_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
�
Mgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_48Generator/last_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:�*
T0
�
Xgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpL^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulN^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
`gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityKgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/MulY^gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*^
_classT
RPloc:@gradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:�*
T0
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
9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1Shape9Generator/third_layer/batch_normalization/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
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
Ggradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs7gradients_1/Generator/third_layer/leaky_relu_grad/Shape9gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8gradients_1/Generator/third_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqualUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/third_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/third_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/third_layer/leaky_relu_grad/zerosUgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_1Sum:gradients_1/Generator/third_layer/leaky_relu_grad/Select_1Igradients_1/Generator/third_layer/leaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
;gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/third_layer/leaky_relu_grad/Sum_19gradients_1/Generator/third_layer/leaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Bgradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_depsNoOp:^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape<^gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1
�
Jgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependencyIdentity9gradients_1/Generator/third_layer/leaky_relu_grad/ReshapeC^gradients_1/Generator/third_layer/leaky_relu_grad/tuple/group_deps*L
_classB
@>loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������*
T0
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
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mul&Generator/third_layer/leaky_relu/alphaJgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/third_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/third_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
gradients_1/AddN_5AddNLgradients_1/Generator/third_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/third_layer/leaky_relu/mul_grad/tuple/control_dependency_1*
T0*N
_classD
B@loc:@gradients_1/Generator/third_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape9Generator/third_layer/batch_normalization/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
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
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_5`gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_5bgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Tgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapePgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
Tshape0*
_output_shapes	
:�*
T0
�
[gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpS^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/ReshapeU^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*e
_class[
YWloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0
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
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
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
Rgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeReshapeNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/SumPgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1Mul-Generator/third_layer/fully_connected/BiasAddcgradients_1/Generator/third_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1\^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
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
Ygradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/group_deps*]
_classS
QOloc:@gradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/MulMulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_17Generator/third_layer/batch_normalization/batchnorm/mul*
T0*
_output_shapes	
:�
�
Pgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Mulcgradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1:Generator/third_layer/batch_normalization/moving_mean/read*
_output_shapes	
:�*
T0
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
Dgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/third_layer/fully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
Vgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityDgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMulO^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Xgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1O^gradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients_1/Generator/third_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
gradients_1/AddN_6AddNegradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1egradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N
�
Lgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_64Generator/third_layer/batch_normalization/gamma/read*
_output_shapes	
:�*
T0
�
Ngradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_69Generator/third_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:�*
T0
�
Ygradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_depsNoOpM^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulO^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1
�
agradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependencyIdentityLgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/MulZ^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*_
_classU
SQloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul*
_output_shapes	
:�*
T0
�
cgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1IdentityNgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1Z^gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
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
8gradients_1/Generator/second_layer/leaky_relu_grad/zerosFill:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_2>gradients_1/Generator/second_layer/leaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:����������*
T0
�
?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqualGreaterEqual%Generator/second_layer/leaky_relu/mul:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
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
;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Select?gradients_1/Generator/second_layer/leaky_relu_grad/GreaterEqual8gradients_1/Generator/second_layer/leaky_relu_grad/zerosVgradients_1/Generator/third_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
6gradients_1/Generator/second_layer/leaky_relu_grad/SumSum9gradients_1/Generator/second_layer/leaky_relu_grad/SelectHgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeReshape6gradients_1/Generator/second_layer/leaky_relu_grad/Sum8gradients_1/Generator/second_layer/leaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1Sum;gradients_1/Generator/second_layer/leaky_relu_grad/Select_1Jgradients_1/Generator/second_layer/leaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1Reshape8gradients_1/Generator/second_layer/leaky_relu_grad/Sum_1:gradients_1/Generator/second_layer/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Cgradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_depsNoOp;^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape=^gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1
�
Kgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependencyIdentity:gradients_1/Generator/second_layer/leaky_relu_grad/ReshapeD^gradients_1/Generator/second_layer/leaky_relu_grad/tuple/group_deps*M
_classC
A?loc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape*(
_output_shapes
:����������*
T0
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
:gradients_1/Generator/second_layer/leaky_relu/mul_grad/MulMulKgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency:Generator/second_layer/batch_normalization/batchnorm/add_1*(
_output_shapes
:����������*
T0
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
@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1Reshape<gradients_1/Generator/second_layer/leaky_relu/mul_grad/Sum_1>gradients_1/Generator/second_layer/leaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ggradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_depsNoOp?^gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeA^gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1
�
Ogradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity>gradients_1/Generator/second_layer/leaky_relu/mul_grad/ReshapeH^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1H^gradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/group_deps*S
_classI
GEloc:@gradients_1/Generator/second_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
gradients_1/AddN_7AddNMgradients_1/Generator/second_layer/leaky_relu_grad/tuple/control_dependency_1Qgradients_1/Generator/second_layer/leaky_relu/mul_grad/tuple/control_dependency_1*O
_classE
CAloc:@gradients_1/Generator/second_layer/leaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������*
T0
�
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ShapeShape:Generator/second_layer/batch_normalization/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
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
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/SumSumgradients_1/AddN_7agradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs*
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
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_7cgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Shape_1*
Tshape0*
_output_shapes	
:�*
T0
�
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/group_deps*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
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
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/MulMuldgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency8Generator/second_layer/batch_normalization/batchnorm/mul*
T0*(
_output_shapes
:����������
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
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1SumQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Mul_1cgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ugradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1ReshapeQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Sum_1Sgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
\gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_depsNoOpT^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/ReshapeV^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyIdentitySgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape
�
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1
�
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/NegNegfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
Zgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_depsNoOpg^gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1N^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
�
bgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependencyIdentityfgradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/tuple/control_dependency_1[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/add_1_grad/Reshape_1
�
dgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1IdentityMgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg[^gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*`
_classV
TRloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/Neg
�
Kgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGraddgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency*
_output_shapes	
:�*
T0*
data_formatNHWC
�
Pgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOpe^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyL^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Xgradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitydgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependencyQ^gradients_1/Generator/second_layer/fully_connected/BiasAdd_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������*
T0
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
Qgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1Muldgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency_1;Generator/second_layer/batch_normalization/moving_mean/read*
_output_shapes	
:�*
T0
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
fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1]^gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*d
_classZ
XVloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/Mul_1
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
Wgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependencyIdentityEgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMulP^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*X
_classN
LJloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
Ygradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityGgradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1P^gradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/group_deps*Z
_classP
NLloc:@gradients_1/Generator/second_layer/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
gradients_1/AddN_8AddNfgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/tuple/control_dependency_1fgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_2_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0*h
_class^
\Zloc:@gradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_1_grad/Reshape_1*
N
�
Mgradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/MulMulgradients_1/AddN_85Generator/second_layer/batch_normalization/gamma/read*
T0*
_output_shapes	
:�
�
Ogradients_1/Generator/second_layer/batch_normalization/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_8:Generator/second_layer/batch_normalization/batchnorm/Rsqrt*
_output_shapes	
:�*
T0
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
9gradients_1/Generator/first_layer/leaky_relu_grad/Shape_2ShapeWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
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
8gradients_1/Generator/first_layer/leaky_relu_grad/SelectSelect>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqualWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency7gradients_1/Generator/first_layer/leaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
:gradients_1/Generator/first_layer/leaky_relu_grad/Select_1Select>gradients_1/Generator/first_layer/leaky_relu_grad/GreaterEqual7gradients_1/Generator/first_layer/leaky_relu_grad/zerosWgradients_1/Generator/second_layer/fully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
;gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1Reshape7gradients_1/Generator/first_layer/leaky_relu_grad/Sum_19gradients_1/Generator/first_layer/leaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
�
=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1Shape-Generator/first_layer/fully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
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
9gradients_1/Generator/first_layer/leaky_relu/mul_grad/SumSum9gradients_1/Generator/first_layer/leaky_relu/mul_grad/MulKgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1Sum;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Mul_1Mgradients_1/Generator/first_layer/leaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1Reshape;gradients_1/Generator/first_layer/leaky_relu/mul_grad/Sum_1=gradients_1/Generator/first_layer/leaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Fgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_depsNoOp>^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape@^gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1
�
Ngradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependencyIdentity=gradients_1/Generator/first_layer/leaky_relu/mul_grad/ReshapeG^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*P
_classF
DBloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
�
Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1Identity?gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1G^gradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/Generator/first_layer/leaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_9AddNLgradients_1/Generator/first_layer/leaky_relu_grad/tuple/control_dependency_1Pgradients_1/Generator/first_layer/leaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*
N
�
Jgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_9*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ogradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_9K^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad
�
Wgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_9P^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/Generator/first_layer/leaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Ygradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGradP^gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/group_deps*]
_classS
QOloc:@gradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
Dgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMulMatMulWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency1Generator/first_layer/fully_connected/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
�
Fgradients_1/Generator/first_layer/fully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise_inWgradients_1/Generator/first_layer/fully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d�*
transpose_a(*
transpose_b( *
T0
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
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
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
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
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
8Generator/first_layer/fully_connected/kernel/Adam/AssignAssign1Generator/first_layer/fully_connected/kernel/AdamCGenerator/first_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
6Generator/first_layer/fully_connected/kernel/Adam/readIdentity1Generator/first_layer/fully_connected/kernel/Adam*
T0*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
_output_shapes
:	d�
�
UGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB"d   �   *
dtype0
�
KGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
valueB
 *    *
dtype0
�
EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillUGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorKGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*

index_type0*
_output_shapes
:	d�*
T0
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
:Generator/first_layer/fully_connected/kernel/Adam_1/AssignAssign3Generator/first_layer/fully_connected/kernel/Adam_1EGenerator/first_layer/fully_connected/kernel/Adam_1/Initializer/zeros*?
_class5
31loc:@Generator/first_layer/fully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0
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
6Generator/first_layer/fully_connected/bias/Adam/AssignAssign/Generator/first_layer/fully_connected/bias/AdamAGenerator/first_layer/fully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(
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
8Generator/first_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/first_layer/fully_connected/bias/Adam_1CGenerator/first_layer/fully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
validate_shape(
�
6Generator/first_layer/fully_connected/bias/Adam_1/readIdentity1Generator/first_layer/fully_connected/bias/Adam_1*=
_class3
1/loc:@Generator/first_layer/fully_connected/bias*
_output_shapes	
:�*
T0
�
TGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
valueB"�      *
dtype0
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
DGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zerosFillTGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorJGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
2Generator/second_layer/fully_connected/kernel/Adam
VariableV2* 
_output_shapes
:
��*
shared_name *@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0
�
9Generator/second_layer/fully_connected/kernel/Adam/AssignAssign2Generator/second_layer/fully_connected/kernel/AdamDGenerator/second_layer/fully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
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
;Generator/second_layer/fully_connected/kernel/Adam_1/AssignAssign4Generator/second_layer/fully_connected/kernel/Adam_1FGenerator/second_layer/fully_connected/kernel/Adam_1/Initializer/zeros*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
9Generator/second_layer/fully_connected/kernel/Adam_1/readIdentity4Generator/second_layer/fully_connected/kernel/Adam_1*@
_class6
42loc:@Generator/second_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0
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
VariableV2*
_output_shapes	
:�*
shared_name *>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
	container *
shape:�*
dtype0
�
7Generator/second_layer/fully_connected/bias/Adam/AssignAssign0Generator/second_layer/fully_connected/bias/AdamBGenerator/second_layer/fully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(
�
5Generator/second_layer/fully_connected/bias/Adam/readIdentity0Generator/second_layer/fully_connected/bias/Adam*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
_output_shapes	
:�*
T0
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
9Generator/second_layer/fully_connected/bias/Adam_1/AssignAssign2Generator/second_layer/fully_connected/bias/Adam_1DGenerator/second_layer/fully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*>
_class4
20loc:@Generator/second_layer/fully_connected/bias*
validate_shape(
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
VariableV2*
_output_shapes	
:�*
shared_name *C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
	container *
shape:�*
dtype0
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
>Generator/second_layer/batch_normalization/gamma/Adam_1/AssignAssign7Generator/second_layer/batch_normalization/gamma/Adam_1IGenerator/second_layer/batch_normalization/gamma/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
validate_shape(
�
<Generator/second_layer/batch_normalization/gamma/Adam_1/readIdentity7Generator/second_layer/batch_normalization/gamma/Adam_1*
T0*C
_class9
75loc:@Generator/second_layer/batch_normalization/gamma*
_output_shapes	
:�
�
FGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zerosConst*
_output_shapes	
:�*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
valueB�*    *
dtype0
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
;Generator/second_layer/batch_normalization/beta/Adam/AssignAssign4Generator/second_layer/batch_normalization/beta/AdamFGenerator/second_layer/batch_normalization/beta/Adam/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�
�
9Generator/second_layer/batch_normalization/beta/Adam/readIdentity4Generator/second_layer/batch_normalization/beta/Adam*
_output_shapes	
:�*
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta
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
=Generator/second_layer/batch_normalization/beta/Adam_1/AssignAssign6Generator/second_layer/batch_normalization/beta/Adam_1HGenerator/second_layer/batch_normalization/beta/Adam_1/Initializer/zeros*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
UGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB"      *
dtype0
�
KGenerator/third_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*?
_class5
31loc:@Generator/third_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
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
6Generator/third_layer/fully_connected/bias/Adam/AssignAssign/Generator/third_layer/fully_connected/bias/AdamAGenerator/third_layer/fully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(
�
4Generator/third_layer/fully_connected/bias/Adam/readIdentity/Generator/third_layer/fully_connected/bias/Adam*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:�*
T0
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
8Generator/third_layer/fully_connected/bias/Adam_1/AssignAssign1Generator/third_layer/fully_connected/bias/Adam_1CGenerator/third_layer/fully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
validate_shape(
�
6Generator/third_layer/fully_connected/bias/Adam_1/readIdentity1Generator/third_layer/fully_connected/bias/Adam_1*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
_output_shapes	
:�*
T0
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
;Generator/third_layer/batch_normalization/gamma/Adam/AssignAssign4Generator/third_layer/batch_normalization/gamma/AdamFGenerator/third_layer/batch_normalization/gamma/Adam/Initializer/zeros*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
9Generator/third_layer/batch_normalization/gamma/Adam/readIdentity4Generator/third_layer/batch_normalization/gamma/Adam*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
_output_shapes	
:�*
T0
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
:Generator/third_layer/batch_normalization/beta/Adam/AssignAssign3Generator/third_layer/batch_normalization/beta/AdamEGenerator/third_layer/batch_normalization/beta/Adam/Initializer/zeros*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
8Generator/third_layer/batch_normalization/beta/Adam/readIdentity3Generator/third_layer/batch_normalization/beta/Adam*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:�*
T0
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
<Generator/third_layer/batch_normalization/beta/Adam_1/AssignAssign5Generator/third_layer/batch_normalization/beta/Adam_1GGenerator/third_layer/batch_normalization/beta/Adam_1/Initializer/zeros*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
:Generator/third_layer/batch_normalization/beta/Adam_1/readIdentity5Generator/third_layer/batch_normalization/beta/Adam_1*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
_output_shapes	
:�*
T0
�
RGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0
�
HGenerator/last_layer/fully_connected/kernel/Adam/Initializer/zeros/ConstConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
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
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container 
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
TGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
JGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
valueB
 *    *
dtype0
�
DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zerosFillTGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorJGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros/Const*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
2Generator/last_layer/fully_connected/kernel/Adam_1
VariableV2*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
9Generator/last_layer/fully_connected/kernel/Adam_1/AssignAssign2Generator/last_layer/fully_connected/kernel/Adam_1DGenerator/last_layer/fully_connected/kernel/Adam_1/Initializer/zeros*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
7Generator/last_layer/fully_connected/kernel/Adam_1/readIdentity2Generator/last_layer/fully_connected/kernel/Adam_1*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel* 
_output_shapes
:
��*
T0
�
PGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
FGenerator/last_layer/fully_connected/bias/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0
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
5Generator/last_layer/fully_connected/bias/Adam/AssignAssign.Generator/last_layer/fully_connected/bias/Adam@Generator/last_layer/fully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
3Generator/last_layer/fully_connected/bias/Adam/readIdentity.Generator/last_layer/fully_connected/bias/Adam*
_output_shapes	
:�*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
�
RGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
HGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
valueB
 *    *
dtype0
�
BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zerosFillRGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorHGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros/Const*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*

index_type0*
_output_shapes	
:�*
T0
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
7Generator/last_layer/fully_connected/bias/Adam_1/AssignAssign0Generator/last_layer/fully_connected/bias/Adam_1BGenerator/last_layer/fully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
5Generator/last_layer/fully_connected/bias/Adam_1/readIdentity0Generator/last_layer/fully_connected/bias/Adam_1*
_output_shapes	
:�*
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias
�
UGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/shape_as_tensorConst*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB:�*
dtype0*
_output_shapes
:
�
KGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
valueB
 *    *
dtype0
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
:Generator/last_layer/batch_normalization/gamma/Adam/AssignAssign3Generator/last_layer/batch_normalization/gamma/AdamEGenerator/last_layer/batch_normalization/gamma/Adam/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
validate_shape(*
_output_shapes	
:�
�
8Generator/last_layer/batch_normalization/gamma/Adam/readIdentity3Generator/last_layer/batch_normalization/gamma/Adam*
_output_shapes	
:�*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma
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
GGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zerosFillWGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/shape_as_tensorMGenerator/last_layer/batch_normalization/gamma/Adam_1/Initializer/zeros/Const*
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*

index_type0*
_output_shapes	
:�
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
DGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zerosFillTGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/shape_as_tensorJGenerator/last_layer/batch_normalization/beta/Adam/Initializer/zeros/Const*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0*
_output_shapes	
:�
�
2Generator/last_layer/batch_normalization/beta/Adam
VariableV2*
_output_shapes	
:�*
shared_name *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
	container *
shape:�*
dtype0
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
7Generator/last_layer/batch_normalization/beta/Adam/readIdentity2Generator/last_layer/batch_normalization/beta/Adam*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
_output_shapes	
:�*
T0
�
VGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB:�*
dtype0*
_output_shapes
:
�
LGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
valueB
 *    *
dtype0
�
FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zerosFillVGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/shape_as_tensorLGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros/Const*
_output_shapes	
:�*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*

index_type0
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
;Generator/last_layer/batch_normalization/beta/Adam_1/AssignAssign4Generator/last_layer/batch_normalization/beta/Adam_1FGenerator/last_layer/batch_normalization/beta/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
validate_shape(
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
8Generator/fake_image/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
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
VariableV2*
shared_name *.
_class$
" loc:@Generator/fake_image/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
'Generator/fake_image/kernel/Adam/AssignAssign Generator/fake_image/kernel/Adam2Generator/fake_image/kernel/Adam/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(
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
:Generator/fake_image/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *.
_class$
" loc:@Generator/fake_image/kernel*
valueB
 *    *
dtype0
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
)Generator/fake_image/kernel/Adam_1/AssignAssign"Generator/fake_image/kernel/Adam_14Generator/fake_image/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@Generator/fake_image/kernel*
validate_shape(* 
_output_shapes
:
��
�
'Generator/fake_image/kernel/Adam_1/readIdentity"Generator/fake_image/kernel/Adam_1*.
_class$
" loc:@Generator/fake_image/kernel* 
_output_shapes
:
��*
T0
�
0Generator/fake_image/bias/Adam/Initializer/zerosConst*
_output_shapes	
:�*,
_class"
 loc:@Generator/fake_image/bias*
valueB�*    *
dtype0
�
Generator/fake_image/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *,
_class"
 loc:@Generator/fake_image/bias*
	container *
shape:�
�
%Generator/fake_image/bias/Adam/AssignAssignGenerator/fake_image/bias/Adam0Generator/fake_image/bias/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
�
#Generator/fake_image/bias/Adam/readIdentityGenerator/fake_image/bias/Adam*
_output_shapes	
:�*
T0*,
_class"
 loc:@Generator/fake_image/bias
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
%Generator/fake_image/bias/Adam_1/readIdentity Generator/fake_image/bias/Adam_1*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes	
:�*
T0
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
Adam_1/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
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
GAdam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdam	ApplyAdam/Generator/second_layer/batch_normalization/beta4Generator/second_layer/batch_normalization/beta/Adam6Generator/second_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/second_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes	
:�*
use_locking( *
T0*B
_class8
64loc:@Generator/second_layer/batch_normalization/beta*
use_nesterov( 
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
BAdam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdam	ApplyAdam*Generator/third_layer/fully_connected/bias/Generator/third_layer/fully_connected/bias/Adam1Generator/third_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonYgradients_1/Generator/third_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*=
_class3
1/loc:@Generator/third_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
GAdam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam/Generator/third_layer/batch_normalization/gamma4Generator/third_layer/batch_normalization/gamma/Adam6Generator/third_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/third_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*B
_class8
64loc:@Generator/third_layer/batch_normalization/gamma*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
FAdam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdam	ApplyAdam.Generator/third_layer/batch_normalization/beta3Generator/third_layer/batch_normalization/beta/Adam5Generator/third_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*A
_class7
53loc:@Generator/third_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
CAdam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdam	ApplyAdam+Generator/last_layer/fully_connected/kernel0Generator/last_layer/fully_connected/kernel/Adam2Generator/last_layer/fully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonWgradients_1/Generator/last_layer/fully_connected/MatMul_grad/tuple/control_dependency_1*>
_class4
20loc:@Generator/last_layer/fully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0
�
AAdam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdam	ApplyAdam)Generator/last_layer/fully_connected/bias.Generator/last_layer/fully_connected/bias/Adam0Generator/last_layer/fully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonXgradients_1/Generator/last_layer/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*<
_class2
0.loc:@Generator/last_layer/fully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
FAdam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdam	ApplyAdam.Generator/last_layer/batch_normalization/gamma3Generator/last_layer/batch_normalization/gamma/Adam5Generator/last_layer/batch_normalization/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/last_layer/batch_normalization/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:�*
use_locking( *
T0*A
_class7
53loc:@Generator/last_layer/batch_normalization/gamma*
use_nesterov( 
�
EAdam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdam	ApplyAdam-Generator/last_layer/batch_normalization/beta2Generator/last_layer/batch_normalization/beta/Adam4Generator/last_layer/batch_normalization/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon`gradients_1/Generator/last_layer/batch_normalization/batchnorm/sub_grad/tuple/control_dependency*@
_class6
42loc:@Generator/last_layer/batch_normalization/beta*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
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
1Adam_1/update_Generator/fake_image/bias/ApplyAdam	ApplyAdamGenerator/fake_image/biasGenerator/fake_image/bias/Adam Generator/fake_image/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/Generator/fake_image/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
use_nesterov( *
_output_shapes	
:�
�	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*,
_class"
 loc:@Generator/fake_image/bias
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
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta22^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam*
T0*,
_class"
 loc:@Generator/fake_image/bias*
_output_shapes
: 
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@Generator/fake_image/bias*
validate_shape(
�	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_12^Adam_1/update_Generator/fake_image/bias/ApplyAdam4^Adam_1/update_Generator/fake_image/kernel/ApplyAdamC^Adam_1/update_Generator/first_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/first_layer/fully_connected/kernel/ApplyAdamF^Adam_1/update_Generator/last_layer/batch_normalization/beta/ApplyAdamG^Adam_1/update_Generator/last_layer/batch_normalization/gamma/ApplyAdamB^Adam_1/update_Generator/last_layer/fully_connected/bias/ApplyAdamD^Adam_1/update_Generator/last_layer/fully_connected/kernel/ApplyAdamH^Adam_1/update_Generator/second_layer/batch_normalization/beta/ApplyAdamI^Adam_1/update_Generator/second_layer/batch_normalization/gamma/ApplyAdamD^Adam_1/update_Generator/second_layer/fully_connected/bias/ApplyAdamF^Adam_1/update_Generator/second_layer/fully_connected/kernel/ApplyAdamG^Adam_1/update_Generator/third_layer/batch_normalization/beta/ApplyAdamH^Adam_1/update_Generator/third_layer/batch_normalization/gamma/ApplyAdamC^Adam_1/update_Generator/third_layer/fully_connected/bias/ApplyAdamE^Adam_1/update_Generator/third_layer/fully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
_output_shapes
: *
N""
train_op

Adam
Adam_1"��
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
Discriminator/prob/bias:0Discriminator/prob/bias/AssignDiscriminator/prob/bias/read:02+Discriminator/prob/bias/Initializer/zeros:08�9���       �N�	98�����A*�
w
discriminator_loss*a	   @���?   @���?      �?!   @���?)��>Y�^@23?��|�?�E̟���?�������:              �?        
s
generator_loss*a	   @i%�?   @i%�?      �?!   @i%�?) Y|��]�?2+Se*8�?uo�p�?�������:              �?        ��!<�       �{�	@������A(*�
w
discriminator_loss*a	   @�� @   @�� @      �?!   @�� @) 9�[|@2ܔ�.�u�?��tM@�������:              �?        
s
generator_loss*a	   ��x�?   ��x�?      �?!   ��x�?) �S�S�?2\l�9�?+Se*8�?�������:              �?        Z�nx�       �{�	������AP*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@r//�n�?2cI���?�P�1���?�������:              �?        
s
generator_loss*a	   `���?   `���?      �?!   `���?) ��b��?2W�i�b�?��Z%��?�������:              �?        @�<c�       �{�	N�D����Ax*�
w
discriminator_loss*a	   �>��?   �>��?      �?!   �>��?) �gz�u?2��]$A�?�{ �ǳ�?�������:              �?        
s
generator_loss*a	    �0@    �0@      �?!    �0@)@�y�y9@2!��v�@زv�5f@�������:              �?        ���       b�D�	?H�����A�*�
w
discriminator_loss*a	   �K��?   �K��?      �?!   �K��?)�P�M�X�?28/�C�ַ?%g�cE9�?�������:              �?        
s
generator_loss*a	   ��3@   ��3@      �?!   ��3@) by��l.@2�DK��@{2�.��@�������:              �?        �tǐ�       b�D�	�L�����A�*�
w
discriminator_loss*a	    ���?    ���?      �?!    ���?)@`N6��Z?2`��a�8�?�/�*>�?�������:              �?        
s
generator_loss*a	   �"�@   �"�@      �?!   �"�@)@���7@2!��v�@زv�5f@�������:              �?        ��       b�D�	%�����A�*�
w
discriminator_loss*a	    �9�?    �9�?      �?!    �9�?) @�7zw?2��]$A�?�{ �ǳ�?�������:              �?        
s
generator_loss*a	   `DR@   `DR@      �?!   `DR@)@2�a	U7@2!��v�@زv�5f@�������:              �?        �T��       b�D�	J����A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)��V%2k?2���g��?I���?�������:              �?        
s
generator_loss*a	   `�@   `�@      �?!   `�@)@j9z1@2�DK��@{2�.��@�������:              �?        M�-��       b�D�	ظ�����A�*�
w
discriminator_loss*a	    ^�?    ^�?      �?!    ^�?) �R7��F?2}Y�4j�?��<�A��?�������:              �?        
s
generator_loss*a	   @�@   @�@      �?!   @�@)�ؽ��`A@2��h:np@S���߮@�������:              �?        �L�       b�D�	�������A�*�
w
discriminator_loss*a	   @2�?   @2�?      �?!   @2�?) �?_7?2�Rc�ݒ?^�S���?�������:              �?        
s
generator_loss*a	   ���@   ���@      �?!   ���@) �Z@@2��h:np@S���߮@�������:              �?        �݆0�       b�D�	�b)����A�*�
w
discriminator_loss*a	   @:��?   @:��?      �?!   @:��?) �5�6?2�Rc�ݒ?^�S���?�������:              �?        
s
generator_loss*a	    �3@    �3@      �?!    �3@) ����>@2زv�5f@��h:np@�������:              �?        vm���       b�D�	-Ju����A�*�
w
discriminator_loss*a	   ��u�?   ��u�?      �?!   ��u�?) )'4��?2-Ա�L�?eiS�m�?�������:              �?        
s
generator_loss*a	   ���@   ���@      �?!   ���@) %}v�qC@2S���߮@)����&@�������:              �?        �����       b�D�	)'�����A�*�
w
discriminator_loss*a	    -��?    -��?      �?!    -��?) H߽Ho?2I���?����iH�?�������:              �?        
s
generator_loss*a	   ��5@   ��5@      �?!   ��5@)��n�P"@2h�5�@�Š)U	@�������:              �?        R:���       b�D�	�t����A�*�
w
discriminator_loss*a	   ��=�?   ��=�?      �?!   ��=�?) $�/O�T?2�/��?�uS��a�?�������:              �?        
s
generator_loss*a	    %�@    %�@      �?!    %�@) �5�2@2{2�.��@!��v�@�������:              �?        '��>�       b�D�	�h����A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �s�5?2���&�?�Rc�ݒ?�������:              �?        
s
generator_loss*a	   @R�@   @R�@      �?!   @R�@)�h���_A@2��h:np@S���߮@�������:              �?        {*���       b�D�	rq�����A�*�
w
discriminator_loss*a	   ��=�?   ��=�?      �?!   ��=�?) ��^T�?2����=��?���J�\�?�������:              �?        
s
generator_loss*a	    J�@    J�@      �?!    J�@)  ki6J@2)����&@a/5L��@�������:              �?        &C*��       b�D�	1�1����A�*�
w
discriminator_loss*a	   �F��?   �F��?      �?!   �F��?)�l��pR$?2eiS�m�?#�+(�ŉ?�������:              �?        
s
generator_loss*a	   �y�@   �y�@      �?!   �y�@) ,�A��F@2S���߮@)����&@�������:              �?        9t�d�       b�D�	������A�*�
w
discriminator_loss*a	   ����?   ����?      �?!   ����?) ^�'�'?2#�+(�ŉ?�7c_XY�?�������:              �?        
s
generator_loss*a	    �+@    �+@      �?!    �+@) ��G�\N@2a/5L��@v@�5m @�������:              �?        ��im�       b�D�	�!����A�*�
w
discriminator_loss*a	   `p?�?   `p?�?      �?!   `p?�?)@B	H�?2>	� �?����=��?�������:              �?        
s
generator_loss*a	   ��@   ��@      �?!   ��@) �`{�F@2S���߮@)����&@�������:              �?        ̽��       b�D�	�`t����A�*�
w
discriminator_loss*a	   ��ǁ?   ��ǁ?      �?!   ��ǁ?)@
/w��?2����=��?���J�\�?�������:              �?        
s
generator_loss*a	   ��k @   ��k @      �?!   ��k @) D���P@2a/5L��@v@�5m @�������:              �?        �ڄ��       b�D�	.������A�*�
w
discriminator_loss*a	    <�?    <�?      �?!    <�?)  	�}?2����=��?���J�\�?�������:              �?        
s
generator_loss*a	   �2b!@   �2b!@      �?!   �2b!@)@.�@�R@2v@�5m @��@�"@�������:              �?        U���       b�D�	�KD����A�*�
w
discriminator_loss*a	   �7w�?   �7w�?      �?!   �7w�?) A
��O5?2���&�?�Rc�ݒ?�������:              �?        
s
generator_loss*a	    Ǎ@    Ǎ@      �?!    Ǎ@) ���gD@2S���߮@)����&@�������:              �?        i����       �N�	�������A*�
w
discriminator_loss*a	   �ʯ�?   �ʯ�?      �?!   �ʯ�?)@r�9?2���J�\�?-Ա�L�?�������:              �?        
s
generator_loss*a	   @�@   @�@      �?!   @�@)������F@2S���߮@)����&@�������:              �?        ���W�       �{�	(������A(*�
w
discriminator_loss*a	   ��>�?   ��>�?      �?!   ��>�?) 9Y��5?2���J�\�?-Ա�L�?�������:              �?        
s
generator_loss*a	    ��@    ��@      �?!    ��@)  �J�VA@2��h:np@S���߮@�������:              �?        �����       �{�	OQ����AP*�
w
discriminator_loss*a	    ���?    ���?      �?!    ���?)  �D7�!?2eiS�m�?#�+(�ŉ?�������:              �?        
s
generator_loss*a	    �_@    �_@      �?!    �_@) H��A@2��h:np@S���߮@�������:              �?        ��y�       �{�	������Ax*�
w
discriminator_loss*a	   �#H?   �#H?      �?!   �#H?) b�gg�?2���T}?>	� �?�������:              �?        
s
generator_loss*a	   `��@   `��@      �?!   `��@) 7̾EF@2S���߮@)����&@�������:              �?        ����       b�D�	'
����A�*�
w
discriminator_loss*a	   ���{?   ���{?      �?!   ���{?) ��ۅ�?2o��5sz?���T}?�������:              �?        
s
generator_loss*a	   ��`@   ��`@      �?!   ��`@) z�C�B@2��h:np@S���߮@�������:              �?        2C���       b�D�	A������A�*�
w
discriminator_loss*a	   �E�?   �E�?      �?!   �E�?)���@�L?2��<�A��?�v��ab�?�������:              �?        
s
generator_loss*a	   �x�@   �x�@      �?!   �x�@) ��H�Q1@2�DK��@{2�.��@�������:              �?        K��A�       b�D�	�������A�*�
w
discriminator_loss*a	   ����?   ����?      �?!   ����?) ����a?2�/�*>�?�g���w�?�������:              �?        
s
generator_loss*a	    �-@    �-@      �?!    �-@) Ή	<@2زv�5f@��h:np@�������:              �?        �B(�       b�D�	�6T����A�*�
w
discriminator_loss*a	   �~�?   �~�?      �?!   �~�?) $��S6?2�Rc�ݒ?^�S���?�������:              �?        
s
generator_loss*a	   @�@   @�@      �?!   @�@) I���5@2!��v�@زv�5f@�������:              �?        �4���       b�D�	C3�����A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) @���:>?2^�S���?�"�uԖ?�������:              �?        
s
generator_loss*a	   �� @   �� @      �?!   �� @)@��,�?Q@2v@�5m @��@�"@�������:              �?        }_ď�       b�D�	�+����A�*�
w
discriminator_loss*a	   ��R�?   ��R�?      �?!   ��R�?) y�R�U?2����=��?���J�\�?�������:              �?        
s
generator_loss*a	   @�1@   @�1@      �?!   @�1@) Ѫ�rMr@2�x�a0@�����1@�������:              �?        ��מ�       b�D�	������A�*�
w
discriminator_loss*a	   `4�z?   `4�z?      �?!   `4�z?) �u\B?2o��5sz?���T}?�������:              �?        
s
generator_loss*a	    T*@    T*@      �?!    T*@) �\��?B@2��h:np@S���߮@�������:              �?        Z}���       b�D�	g�����A�*�
w
discriminator_loss*a	   ��Wy?   ��Wy?      �?!   ��Wy?) B ?2*QH�x?o��5sz?�������:              �?        
s
generator_loss*a	   �V�!@   �V�!@      �?!   �V�!@)@�0���S@2v@�5m @��@�"@�������:              �?        >mP��       b�D�	#�z����A�*�
w
discriminator_loss*a	    +�m?    +�m?      �?!    +�m?) T0���>2ߤ�(g%k?�N�W�m?�������:              �?        
s
generator_loss*a	   @�� @   @�� @      �?!   @�� @) )�h�Q@2v@�5m @��@�"@�������:              �?        wY��       b�D�	�������A�*�
w
discriminator_loss*a	   �eTd?   �eTd?      �?!   �eTd?)@z�=���>2���%��b?5Ucv0ed?�������:              �?        
s
generator_loss*a	    ��!@    ��!@      �?!    ��!@) ��Y�S@2v@�5m @��@�"@�������:              �?        ����       b�D�	ȱb����A�*�
w
discriminator_loss*a	   @}�W?   @}�W?      �?!   @}�W?)�<��Q��>2ܗ�SsW?��bB�SY?�������:              �?        
s
generator_loss*a	   @8S"@   @8S"@      �?!   @8S"@) �]h��T@2��@�"@�ՠ�M�#@�������:              �?        )�G�       b�D�	h�����A�*�
w
discriminator_loss*a	   ��P?   ��P?      �?!   ��P?) �z��>2k�1^�sO?nK���LQ?�������:              �?        
s
generator_loss*a	   @6�"@   @6�"@      �?!   @6�"@) ��m�8V@2��@�"@�ՠ�M�#@�������:              �?        ����       b�D�	��O����A�*�
w
discriminator_loss*a	   @��T?   @��T?      �?!   @��T?) !5H�>2�lDZrS?<DKc��T?�������:              �?        
s
generator_loss*a	   ��N#@   ��N#@      �?!   ��N#@) ��{LW@2��@�"@�ՠ�M�#@�������:              �?        ����       b�D�	W������A�*�
w
discriminator_loss*a	   @K_?   @K_?      �?!   @K_?)���u#�>2E��{��^?�l�P�`?�������:              �?        
s
generator_loss*a	   ���#@   ���#@      �?!   ���#@) D��'X@2��@�"@�ՠ�M�#@�������:              �?        *����       b�D�	^F����A�*�
w
discriminator_loss*a	   ��]?   ��]?      �?!   ��]?)�p=�<|�>2�m9�H�[?E��{��^?�������:              �?        
s
generator_loss*a	   @^�#@   @^�#@      �?!   @^�#@) 1{"D�X@2��@�"@�ՠ�M�#@�������:              �?        a3�2�       b�D�	y������A�*�
w
discriminator_loss*a	   @qzP?   @qzP?      �?!   @qzP?) �1����>2k�1^�sO?nK���LQ?�������:              �?        
s
generator_loss*a	   ��+$@   ��+$@      �?!   ��+$@) ��PnY@2�ՠ�M�#@sQ��"�%@�������:              �?        ?a��       b�D�	"C����A�*�
w
discriminator_loss*a	   `��F?   `��F?      �?!   `��F?) �/6�D�>2a�$��{E?
����G?�������:              �?        
s
generator_loss*a	   �\g$@   �\g$@      �?!   �\g$@) ��Z@2�ՠ�M�#@sQ��"�%@�������:              �?        �U�D�       b�D�	Ξ�����A�*�
w
discriminator_loss*a	   ���F?   ���F?      �?!   ���F?) ��J���>2a�$��{E?
����G?�������:              �?        
s
generator_loss*a	    W�$@    W�$@      �?!    W�$@)@l2�ATZ@2�ՠ�M�#@sQ��"�%@�������:              �?        !hb�       �N�	r1����A*�
w
discriminator_loss*a	   `�y8?   `�y8?      �?!   `�y8?) ��m)��>2uܬ�@8?��%>��:?�������:              �?        
s
generator_loss*a	   ���$@   ���$@      �?!   ���$@)@���Z@2�ՠ�M�#@sQ��"�%@�������:              �?        n���       �{�	������A(*�
w
discriminator_loss*a	   @�?D?   @�?D?      �?!   @�?D?) )
�ߠ�>2�T���C?a�$��{E?�������:              �?        
s
generator_loss*a	   ���$@   ���$@      �?!   ���$@)@Ҧ�&[@2�ՠ�M�#@sQ��"�%@�������:              �?        ù���       �{�	->����AP*�
w
discriminator_loss*a	   �yhQ?   �yhQ?      �?!   �yhQ?)@X 0��>2nK���LQ?�lDZrS?�������:              �?        
s
generator_loss*a	   �%@   �%@      �?!   �%@)@��]'�[@2�ՠ�M�#@sQ��"�%@�������:              �?        ���       �{�	S[�����Ax*�
w
discriminator_loss*a	   �[@?   �[@?      �?!   �[@?) D;˸
�>2d�\D�X=?���#@?�������:              �?        
s
generator_loss*a	   ` =%@   ` =%@      �?!   ` =%@)@��^1\@2�ՠ�M�#@sQ��"�%@�������:              �?        핪��       b�D�	�Q����A�*�
w
discriminator_loss*a	   @�B?   @�B?      �?!   @�B?) ���dŕ>2�!�A?�T���C?�������:              �?        
s
generator_loss*a	   @�p%@   @�p%@      �?!   @�p%@) �^���\@2�ՠ�M�#@sQ��"�%@�������:              �?        +8�"�       b�D�	�������A�*�
w
discriminator_loss*a	   ���G?   ���G?      �?!   ���G?) .M��>2
����G?�qU���I?�������:              �?        
s
generator_loss*a	   �ؗ%@   �ؗ%@      �?!   �ؗ%@) I�:$]@2�ՠ�M�#@sQ��"�%@�������:              �?        ,P)��       b�D�	��i����A�*�
w
discriminator_loss*a	    �A?    �A?      �?!    �A?) ���0�>2���#@?�!�A?�������:              �?        
s
generator_loss*a	   @F�%@   @F�%@      �?!   @F�%@) q<EWS]@2�ՠ�M�#@sQ��"�%@�������:              �?        �j4�       b�D�	K2�����A�*�
w
discriminator_loss*a	   �0XC?   �0XC?      �?!   �0XC?)@L5.Zc�>2�!�A?�T���C?�������:              �?        
s
generator_loss*a	   ���%@   ���%@      �?!   ���%@) DQI��]@2�ՠ�M�#@sQ��"�%@�������:              �?        ��>��       b�D�	������A�*�
w
discriminator_loss*a	    ��G?    ��G?      �?!    ��G?) ���>2
����G?�qU���I?�������:              �?        
s
generator_loss*a	   ���%@   ���%@      �?!   ���%@) i(�^@2sQ��"�%@e��:�(@�������:              �?        Ot?�       b�D�	������A�*�
w
discriminator_loss*a	    )�@?    )�@?      �?!    )�@?) �w���>2���#@?�!�A?�������:              �?        
s
generator_loss*a	   ��&@   ��&@      �?!   ��&@) $9�q^@2sQ��"�%@e��:�(@�������:              �?        e=��       b�D�	$������A�*�
w
discriminator_loss*a	   �5?   �5?      �?!   �5?)@��,�{>2�u�w74?��%�V6?�������:              �?        
s
generator_loss*a	   @\,&@   @\,&@      �?!   @\,&@) �s�x�^@2sQ��"�%@e��:�(@�������:              �?        �,�       b�D�	�G����A�*�
w
discriminator_loss*a	   ��1?   ��1?      �?!   ��1?)@����=r>2��bȬ�0?��82?�������:              �?        
s
generator_loss*a	   ��]&@   ��]&@      �?!   ��]&@) ��2D_@2sQ��"�%@e��:�(@�������:              �?        ��P�       b�D�	[�����A�*�
w
discriminator_loss*a	   �-?   �-?      �?!   �-?) V)��|j>2�7Kaa+?��VlQ.?�������:              �?        
s
generator_loss*a	   ��}&@   ��}&@      �?!   ��}&@) Ds���_@2sQ��"�%@e��:�(@�������:              �?        M���       b�D�	�~x����A�*�
w
discriminator_loss*a	   �Ca;?   �Ca;?      �?!   �Ca;?)�pK�Im�>2��%>��:?d�\D�X=?�������:              �?        
s
generator_loss*a	   ���&@   ���&@      �?!   ���&@) Q`�R�_@2sQ��"�%@e��:�(@�������:              �?        ����       b�D�	�.����A�*�
w
discriminator_loss*a	   @��??   @��??      �?!   @��??)��%kV�>2d�\D�X=?���#@?�������:              �?        
s
generator_loss*a	   �C�&@   �C�&@      �?!   �C�&@)�p��?`@2sQ��"�%@e��:�(@�������:              �?        c�:�       b�D�	Ѯ����A�*�
w
discriminator_loss*a	    �,?    �,?      �?!    �,?) �ƅ��i>2�7Kaa+?��VlQ.?�������:              �?        
s
generator_loss*a	    ��&@    ��&@      �?!    ��&@) �m4�K`@2sQ��"�%@e��:�(@�������:              �?        ���       b�D�	�N����A�*�
w
discriminator_loss*a	    �O$?    �O$?      �?!    �O$?)@xj�n�Y>2�[^:��"?U�4@@�$?�������:              �?        
s
generator_loss*a	   �Y�&@   �Y�&@      �?!   �Y�&@)���$2O`@2sQ��"�%@e��:�(@�������:              �?         t�|�       b�D�	.������A�*�
w
discriminator_loss*a	    -3?    -3?      �?!    -3?)@L"т�v>2��82?�u�w74?�������:              �?        
s
generator_loss*a	   ���&@   ���&@      �?!   ���&@)�d0��`@2sQ��"�%@e��:�(@�������:              �?        �����       b�D�	������A�*�
w
discriminator_loss*a	   @��&?   @��&?      �?!   @��&?) �e���_>2U�4@@�$?+A�F�&?�������:              �?        
s
generator_loss*a	    0'@    0'@      �?!    0'@)  H�e�`@2sQ��"�%@e��:�(@�������:              �?        ����       b�D�	��1����A�*�
w
discriminator_loss*a	   ��>#?   ��>#?      �?!   ��>#?) !^p&W>2�[^:��"?U�4@@�$?�������:              �?        
s
generator_loss*a	   `-'@   `-'@      �?!   `-'@) cE��`@2sQ��"�%@e��:�(@�������:              �?        rnN�       b�D�	�>�����A�*�
w
discriminator_loss*a	   �۝+?   �۝+?      �?!   �۝+?) Y�Vm�g>2�7Kaa+?��VlQ.?�������:              �?        
s
generator_loss*a	   �+&@   �+&@      �?!   �+&@)@�΄g^@2sQ��"�%@e��:�(@�������:              �?        �����       b�D�	Ж}����A�*�
w
discriminator_loss*a	   ��'?   ��'?      �?!   ��'?) �O%�fa>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	   ��%@   ��%@      �?!   ��%@)@N4d�/^@2sQ��"�%@e��:�(@�������:              �?        M���       �N�	�����A*�
w
discriminator_loss*a	   ��'?   ��'?      �?!   ��'?) ��\w�`>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	   ��[&@   ��[&@      �?!   ��[&@) !r��>_@2sQ��"�%@e��:�(@�������:              �?        �h<�       �{�	U������A(*�
w
discriminator_loss*a	    �]?    �]?      �?!    �]?) �����0>2x?�x�?��d�r?�������:              �?        
s
generator_loss*a	    B�&@    B�&@      �?!    B�&@)  ��?8`@2sQ��"�%@e��:�(@�������:              �?        �S�(�       �{�	�}f����AP*�
w
discriminator_loss*a	   ���)?   ���)?      �?!   ���)?)�Ȁ7��d>2I�I�)�(?�7Kaa+?�������:              �?        
s
generator_loss*a	   ��'@   ��'@      �?!   ��'@) B���`@2sQ��"�%@e��:�(@�������:              �?        Y:LL�       �{�	�P����Ax*�
w
discriminator_loss*a	   ��6?   ��6?      �?!   ��6?) ����>2��%�V6?uܬ�@8?�������:              �?        
s
generator_loss*a	   �Aa'@   �Aa'@      �?!   �Aa'@) ���a@2sQ��"�%@e��:�(@�������:              �?        �2�#�       b�D�	�0�����A�*�
w
discriminator_loss*a	    jN?    jN?      �?!    jN?) ���ED>2�vV�R9?��ڋ?�������:              �?        
s
generator_loss*a	   @��'@   @��'@      �?!   @��'@)��Qa_a@2sQ��"�%@e��:�(@�������:              �?        ֠A�       b�D�	�u����A�*�
w
discriminator_loss*a	   ��#?   ��#?      �?!   ��#?) Y�Nn�6>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   @g�'@   @g�'@      �?!   @g�'@)�$m��}a@2sQ��"�%@e��:�(@�������:              �?        4���       b�D�	8�'����A�*�
w
discriminator_loss*a	    �/?    �/?      �?!    �/?) @���w9>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	   @�'@   @�'@      �?!   @�'@)�8!Jk�a@2sQ��"�%@e��:�(@�������:              �?        ���       b�D�	/������A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)�X&���A>2�vV�R9?��ڋ?�������:              �?        
s
generator_loss*a	   @��'@   @��'@      �?!   @��'@)�����a@2sQ��"�%@e��:�(@�������:              �?        +���       b�D�	g�����A�*�
w
discriminator_loss*a	   ���#?   ���#?      �?!   ���#?) a:�VtX>2�[^:��"?U�4@@�$?�������:              �?        
s
generator_loss*a	   �?�'@   �?�'@      �?!   �?�'@) ����a@2sQ��"�%@e��:�(@�������:              �?        eЊ�       b�D�	_c����A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) ˸v,jJ>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	    (@    (@      �?!    (@)  f̕b@2sQ��"�%@e��:�(@�������:              �?        \�/��       b�D�	!�+ ���A�*�
w
discriminator_loss*a	    �E ?    �E ?      �?!    �E ?)@�t��P>2ji6�9�?�S�F !?�������:              �?        
s
generator_loss*a	   ��!(@   ��!(@      �?!   ��!(@) ��j2b@2e��:�(@����t*@�������:              �?        U��c�       b�D�	&%���A�*�
w
discriminator_loss*a	   `�9?   `�9?      �?!   `�9?) ��=�>2uܬ�@8?��%>��:?�������:              �?        
s
generator_loss*a	   ��7(@   ��7(@      �?!   ��7(@) RA)XTb@2e��:�(@����t*@�������:              �?        �k.��       b�D�	������A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@Nb��3>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   ��P(@   ��P(@      �?!   ��P(@) o[�yb@2e��:�(@����t*@�������:              �?        ���r�       b�D�	�����A�*�
w
discriminator_loss*a	   @$G?   @$G?      �?!   @$G?)�!�,>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   �F(@   �F(@      �?!   �F(@) Rץ�jb@2e��:�(@����t*@�������:              �?        jqk�       b�D�	%D^���A�*�
w
discriminator_loss*a	   ���,?   ���,?      �?!   ���,?) �ɰj>2�7Kaa+?��VlQ.?�������:              �?        
s
generator_loss*a	   ��M(@   ��M(@      �?!   ��M(@) �^�cub@2e��:�(@����t*@�������:              �?        ���       b�D�	I�"���A�*�
w
discriminator_loss*a	   �b~%?   �b~%?      �?!   �b~%?) d���\>2U�4@@�$?+A�F�&?�������:              �?        
s
generator_loss*a	    -d(@    -d(@      �?!    -d(@) H}�b@2e��:�(@����t*@�������:              �?        �6���       b�D�	�����A�*�
w
discriminator_loss*a	   @
v?   @
v?      �?!   @
v?)�H���G>2��ڋ?�.�?�������:              �?        
s
generator_loss*a	    1(@    1(@      �?!    1(@) j��%b@2e��:�(@����t*@�������:              �?        �)���       b�D�	�����A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) ����|C>2�vV�R9?��ڋ?�������:              �?        
s
generator_loss*a	   @�H(@   @�H(@      �?!   @�H(@)����vmb@2e��:�(@����t*@�������:              �?        tM���       b�D�	r�q���A�*�
w
discriminator_loss*a	    )�"?    )�"?      �?!    )�"?)@���WcV>2�[^:��"?U�4@@�$?�������:              �?        
s
generator_loss*a	   �m{(@   �m{(@      �?!   �m{(@) �>S �b@2e��:�(@����t*@�������:              �?        ���B�       b�D�	�;���A�*�
w
discriminator_loss*a	   �V?   �V?      �?!   �V?) �X��9>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	   @�(@   @�(@      �?!   @�(@)����H�b@2e��:�(@����t*@�������:              �?        ��bk�       b�D�	�����A�*�
w
discriminator_loss*a	   @�C ?   @�C ?      �?!   @�C ?) ��Ȉ>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   ���(@   ���(@      �?!   ���(@)����W�b@2e��:�(@����t*@�������:              �?        $�ʴ�       b�D�	�'����A�*�
w
discriminator_loss*a	   �\q?   �\q?      �?!   �\q?) �� �GI>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	    K�(@    K�(@      �?!    K�(@) �_�&c@2e��:�(@����t*@�������:              �?        ��_X�       �N�	�,�	���A*�
w
discriminator_loss*a	   `��?   `��?      �?!   `��?) ��b�O@>2�T7��?�vV�R9?�������:              �?        
s
generator_loss*a	   `��(@   `��(@      �?!   `��(@) ��T-c@2e��:�(@����t*@�������:              �?        $*j�       �{�	8t
���A(*�
w
discriminator_loss*a	   �%?   �%?      �?!   �%?)@�a=!�[>2U�4@@�$?+A�F�&?�������:              �?        
s
generator_loss*a	   �:�(@   �:�(@      �?!   �:�(@) R���Tc@2e��:�(@����t*@�������:              �?        �+��       �{�	��F���AP*�
w
discriminator_loss*a	   @�?   @�?      �?!   @�?)� �8N>2ji6�9�?�S�F !?�������:              �?        
s
generator_loss*a	   �@�(@   �@�(@      �?!   �@�(@) j��c@2e��:�(@����t*@�������:              �?        ���       �{�	�~���Ax*�
w
discriminator_loss*a	   �/	?   �/	?      �?!   �/	?) �; ��#>2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	    z	)@    z	)@      �?!    z	)@)  qnіc@2e��:�(@����t*@�������:              �?        ]��       b�D�	������A�*�
w
discriminator_loss*a	   `7?   `7?      �?!   `7?) Ӄ�SpJ>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	    �)@    �)@      �?!    �)@)  
��c@2e��:�(@����t*@�������:              �?        ��,��       b�D�	4�����A�*�
w
discriminator_loss*a	   `1~?   `1~?      �?!   `1~?)@^@�?>2�T7��?�vV�R9?�������:              �?        
s
generator_loss*a	   @�@)@   @�@)@      �?!   @�@)@)�0�؏�c@2e��:�(@����t*@�������:              �?        AU4Y�       b�D�	�`����A�*�
w
discriminator_loss*a	   ��S?   ��S?      �?!   ��S?)�Ȑ�!>26�]��?����?�������:              �?        
s
generator_loss*a	   @yH)@   @yH)@      �?!   @yH)@)�lk���c@2e��:�(@����t*@�������:              �?        T!��       b�D�	9�����A�*�
w
discriminator_loss*a	   `q�?   `q�?      �?!   `q�?) �5���K>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	   �Cb)@   �Cb)@      �?!   �Cb)@) b�5�"d@2e��:�(@����t*@�������:              �?        �|4��       b�D�	fܘ���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@�de6>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   �Xm)@   �Xm)@      �?!   �Xm)@)�$�QP4d@2e��:�(@����t*@�������:              �?        ���<�       b�D�	�v���A�*�
w
discriminator_loss*a	   `�f?   `�f?      �?!   `�f?)@�3#:>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	   �p)@   �p)@      �?!   �p)@) ���8d@2e��:�(@����t*@�������:              �?        6�!R�       b�D�	��T���A�*�
w
discriminator_loss*a	   @o�>   @o�>      �?!   @o�>)��V;&>2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	    ��)@    ��)@      �?!    ��)@) H���dd@2e��:�(@����t*@�������:              �?        
Rus�       b�D�	?�5���A�*�
w
discriminator_loss*a	   ``C?   ``C?      �?!   ``C?) A���,>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   �Ν)@   �Ν)@      �?!   �Ν)@)���E��d@2e��:�(@����t*@�������:              �?        �9�       b�D�	@����A�*�
w
discriminator_loss*a	   @�P?   @�P?      �?!   @�P?) ���?>2�T7��?�vV�R9?�������:              �?        
s
generator_loss*a	    e�)@    e�)@      �?!    e�)@) Ȟ֍�d@2e��:�(@����t*@�������:              �?        � �G�       b�D�	����A�*�
w
discriminator_loss*a	   ���4?   ���4?      �?!   ���4?)@��V${>2�u�w74?��%�V6?�������:              �?        
s
generator_loss*a	   ��K)@   ��K)@      �?!   ��K)@)�h#v��c@2e��:�(@����t*@�������:              �?        ��>�       b�D�	$����A�*�
w
discriminator_loss*a	   @d?�>   @d?�>      �?!   @d?�>) !��N��=2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   �)n)@   �)n)@      �?!   �)n)@) %bH�5d@2e��:�(@����t*@�������:              �?        �q��       b�D�	:����A�*�
w
discriminator_loss*a	    *
?    *
?      �?!    *
?)  O;L%>2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	   �Ē)@   �Ē)@      �?!   �Ē)@)����od@2e��:�(@����t*@�������:              �?        �@j��       b�D�	�E����A�*�
w
discriminator_loss*a	   @R!?   @R!?      �?!   @R!?) ���>26�]��?����?�������:              �?        
s
generator_loss*a	   ��h(@   ��h(@      �?!   ��h(@) R��O�b@2e��:�(@����t*@�������:              �?        �����       b�D�	Pj����A�*�
w
discriminator_loss*a	   @� ?   @� ?      �?!   @� ?) 9��>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   �d�(@   �d�(@      �?!   �d�(@) ��	Dc@2e��:�(@����t*@�������:              �?        9*���       b�D�	����A�*�
w
discriminator_loss*a	   �7�>   �7�>      �?!   �7�>) A���=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   �9')@   �9')@      �?!   �9')@) R��y�c@2e��:�(@����t*@�������:              �?        �@Ke�       b�D�	r���A�*�
w
discriminator_loss*a	   ���$?   ���$?      �?!   ���$?)@~�`�[>2U�4@@�$?+A�F�&?�������:              �?        
s
generator_loss*a	   @�i)@   @�i)@      �?!   @�i)@)�(_�.d@2e��:�(@����t*@�������:              �?        m��       b�D�	�a���A�*�
w
discriminator_loss*a	   �l�?   �l�?      �?!   �l�?)�����M >26�]��?����?�������:              �?        
s
generator_loss*a	   �9�)@   �9�)@      �?!   �9�)@)�8d&�d@2e��:�(@����t*@�������:              �?        �^��       b�D�	8�Q���A�*�
w
discriminator_loss*a	   �Ӹ�>   �Ӹ�>      �?!   �Ӹ�>)@�HNO�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	    ��)@    ��)@      �?!    ��)@)  �־d@2e��:�(@����t*@�������:              �?        7����       �N�	�.���A*�
w
discriminator_loss*a	    \H�>    \H�>      �?!    \H�>) 8�֔�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   ���)@   ���)@      �?!   ���)@)��[<�d@2e��:�(@����t*@�������:              �?        {���       �{�	s�6���A(*�
w
discriminator_loss*a	   ��c�>   ��c�>      �?!   ��c�>)@\,��~�=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	    2�)@    2�)@      �?!    2�)@) ����e@2e��:�(@����t*@�������:              �?        ���       �{�	?Y���AP*�
w
discriminator_loss*a	   ��l"?   ��l"?      �?!   ��l"?)@��&7U>2�S�F !?�[^:��"?�������:              �?        
s
generator_loss*a	    �*@    �*@      �?!    �*@) HR��Ie@2e��:�(@����t*@�������:              �?        3=�,�       �{�	�ki ���Ax*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) ȶ��A>2�vV�R9?��ڋ?�������:              �?        
s
generator_loss*a	   ��0*@   ��0*@      �?!   ��0*@) 2��oe@2e��:�(@����t*@�������:              �?        \��       b�D�	��n!���A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)�h�L~o)>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   @�H*@   @�H*@      �?!   @�H*@)�<��ؖe@2e��:�(@����t*@�������:              �?        H���       b�D�	ԉo"���A�*�
w
discriminator_loss*a	   `+P?   `+P?      �?!   `+P?)@�5��O>2��[�?1��a˲?�������:              �?        
s
generator_loss*a	   ��U*@   ��U*@      �?!   ��U*@) ��4�e@2e��:�(@����t*@�������:              �?        �f]��       b�D�	C�o#���A�*�
w
discriminator_loss*a	   ��o�>   ��o�>      �?!   ��o�>) ��&$��=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   @Ku*@   @Ku*@      �?!   @Ku*@)���6H�e@2����t*@�}h�-@�������:              �?        H��l�       b�D�	�ix$���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  i����=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   �D�*@   �D�*@      �?!   �D�*@) ���f@2����t*@�}h�-@�������:              �?        �����       b�D�	���%���A�*�
w
discriminator_loss*a	   �o^�>   �o^�>      �?!   �o^�>)�@R���=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	    �*@    �*@      �?!    �*@) Hz�C*f@2����t*@�}h�-@�������:              �?        f�4�       b�D�	{u�&���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)��?2�-
>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	    �*@    �*@      �?!    �*@) H۷<@f@2����t*@�}h�-@�������:              �?        �*���       b�D�	�I�'���A�*�
w
discriminator_loss*a	   ���*?   ���*?      �?!   ���*?) Vf��f>2I�I�)�(?�7Kaa+?�������:              �?        
s
generator_loss*a	   @�*@   @�*@      �?!   @�*@)��6[\Yf@2����t*@�}h�-@�������:              �?        �8S=�       b�D�	��(���A�*�
w
discriminator_loss*a	   @&E?   @&E?      �?!   @&E?)��!�E='>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   @b�*@   @b�*@      �?!   @b�*@)��5�Ef@2����t*@�}h�-@�������:              �?        �Ȭ��       b�D�	��)���A�*�
w
discriminator_loss*a	   �h��>   �h��>      �?!   �h��>) ���k�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   `c�*@   `c�*@      �?!   `c�*@) �.��hf@2����t*@�}h�-@�������:              �?        c���       b�D�	��*���A�*�
w
discriminator_loss*a	   �*�?   �*�?      �?!   �*�?)@�b���5>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   ���*@   ���*@      �?!   ���*@)�����f@2����t*@�}h�-@�������:              �?        �\X�       b�D�	�_,���A�*�
w
discriminator_loss*a	   �s�?   �s�?      �?!   �s�?) ���4+>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   @E�*@   @E�*@      �?!   @E�*@)��)iyf@2����t*@�}h�-@�������:              �?        5���       b�D�	�H+-���A�*�
w
discriminator_loss*a	   ��o?   ��o?      �?!   ��o?)�����">2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	    ��*@    ��*@      �?!    ��*@)  ���f@2����t*@�}h�-@�������:              �?        Q? �       b�D�	�I.���A�*�
w
discriminator_loss*a	    �G�>    �G�>      �?!    �G�>)@�I+�M�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   ���*@   ���*@      �?!   ���*@) ���f@2����t*@�}h�-@�������:              �?        ��9�       b�D�	b�\/���A�*�
w
discriminator_loss*a	   �'*�>   �'*�>      �?!   �'*�>) �a\��=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   @��*@   @��*@      �?!   @��*@)���8h�f@2����t*@�}h�-@�������:              �?        ?���       b�D�	lVq0���A�*�
w
discriminator_loss*a	   �@:�>   �@:�>      �?!   �@:�>) �Uu�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   ��6*@   ��6*@      �?!   ��6*@)��ۄFye@2e��:�(@����t*@�������:              �?        [����       b�D�	���1���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  b'e�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	    l�*@    l�*@      �?!    l�*@) �l|S�e@2����t*@�}h�-@�������:              �?        �~k��       b�D�	���2���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@rx�ȱ>26�]��?����?�������:              �?        
s
generator_loss*a	   �%�*@   �%�*@      �?!   �%�*@) �Jthf@2����t*@�}h�-@�������:              �?        D�v�       b�D�	ؔ�3���A�*�
w
discriminator_loss*a	   @O��>   @O��>      �?!   @O��>) �x �=2�f����>��(���>�������:              �?        
s
generator_loss*a	    fq*@    fq*@      �?!    fq*@)  ����e@2e��:�(@����t*@�������:              �?        <7S�       �N�	L�4���A*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@lm/�<�=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   �^�*@   �^�*@      �?!   �^�*@) O= f@2����t*@�}h�-@�������:              �?        �fc��       �{�	�b�5���A(*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ����=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   @0�*@   @0�*@      �?!   @0�*@)���~��f@2����t*@�}h�-@�������:              �?        ,j��       �{�	Ȁ�6���AP*�
w
discriminator_loss*a	   �Q��>   �Q��>      �?!   �Q��>) $/��n�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	    X+@    X+@      �?!    X+@)  ��f@2����t*@�}h�-@�������:              �?        �È��       �{�	�8���Ax*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) A!*�c>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   `�(+@   `�(+@      �?!   `�(+@) Qe�g@2����t*@�}h�-@�������:              �?        ϊC��       b�D�	2U=9���A�*�
w
discriminator_loss*a	    }7?    }7?      �?!    }7?) �0�x�>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	    �=+@    �=+@      �?!    �=+@) �+�h0g@2����t*@�}h�-@�������:              �?        ���       b�D�	��a:���A�*�
w
discriminator_loss*a	   �[��>   �[��>      �?!   �[��>) DK��>�=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	    �]+@    �]+@      �?!    �]+@) kP`gg@2����t*@�}h�-@�������:              �?        x E��       b�D�	��;���A�*�
w
discriminator_loss*a	   �c�>   �c�>      �?!   �c�>)��wC�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	    �l+@    �l+@      �?!    �l+@) Ȅ�g@2����t*@�}h�-@�������:              �?        /�       b�D�	�<���A�*�
w
discriminator_loss*a	    B�>    B�>      �?!    B�>) @�Д��=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   ��y+@   ��y+@      �?!   ��y+@)�0�"e�g@2����t*@�}h�-@�������:              �?        d��i�       b�D�	`��=���A�*�
w
discriminator_loss*a	   @ԍR?   @ԍR?      �?!   @ԍR?) �����>2nK���LQ?�lDZrS?�������:              �?        
s
generator_loss*a	   ��g@   ��g@      �?!   ��g@) B/ <xG@2)����&@a/5L��@�������:              �?        ��l=�       b�D�	ێ ?���A�*�
w
discriminator_loss*a	   �
[M?   �
[M?      �?!   �
[M?)��'��>2IcD���L?k�1^�sO?�������:              �?        
s
generator_loss*a	    �(@    �(@      �?!    �(@) =��J@2)����&@a/5L��@�������:              �?        sD���       b�D�	�.@���A�*�
w
discriminator_loss*a	   �^g?   �^g?      �?!   �^g?) B�@�>2Tw��Nof?P}���h?�������:              �?        
s
generator_loss*a	   @�@   @�@      �?!   @�@)���)E@2S���߮@)����&@�������:              �?        ���       b�D�	�]A���A�*�
w
discriminator_loss*a	   ��_?   ��_?      �?!   ��_?) c�7+�>2E��{��^?�l�P�`?�������:              �?        
s
generator_loss*a	   �n�)@   �n�)@      �?!   �n�)@) ���e@2e��:�(@����t*@�������:              �?        r��,�       b�D�	�}�B���A�*�
w
discriminator_loss*a	   �=N?   �=N?      �?!   �=N?) �WB|>�>2IcD���L?k�1^�sO?�������:              �?        
s
generator_loss*a	   �b�$@   �b�$@      �?!   �b�$@)@3�8�Z@2�ՠ�M�#@sQ��"�%@�������:              �?        k���       b�D�	��C���A�*�
w
discriminator_loss*a	   �y�D?   �y�D?      �?!   �y�D?) q>k�\�>2�T���C?a�$��{E?�������:              �?        
s
generator_loss*a	    a(@    a(@      �?!    a(@) ��$b@2e��:�(@����t*@�������:              �?        �
m�       b�D�	@��D���A�*�
w
discriminator_loss*a	    ��6?    ��6?      �?!    ��6?) ���f�>2��%�V6?uܬ�@8?�������:              �?        
s
generator_loss*a	   @�1$@   @�1$@      �?!   @�1$@) ��-}Y@2�ՠ�M�#@sQ��"�%@�������:              �?        �2��       b�D�	��-F���A�*�
w
discriminator_loss*a	    ��$?    ��$?      �?!    ��$?)@�����Z>2U�4@@�$?+A�F�&?�������:              �?        
s
generator_loss*a	   ��+&@   ��+&@      �?!   ��+&@) dA���^@2sQ��"�%@e��:�(@�������:              �?        A�8$�       b�D�	WcG���A�*�
w
discriminator_loss*a	   �#7?   �#7?      �?!   �#7?)��#����>2��%�V6?uܬ�@8?�������:              �?        
s
generator_loss*a	    �/'@    �/'@      �?!    �/'@) �� �`@2sQ��"�%@e��:�(@�������:              �?        ��x��       b�D�	u��H���A�*�
w
discriminator_loss*a	   `�0?   `�0?      �?!   `�0?)@Y�w7>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   ��(@   ��(@      �?!   ��(@) ��b@2sQ��"�%@e��:�(@�������:              �?        �q�       b�D�	�?�I���A�*�
w
discriminator_loss*a	   �)2?   �)2?      �?!   �)2?)�x��\~,>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   `�a(@   `�a(@      �?!   `�a(@) ����b@2e��:�(@����t*@�������:              �?        ���       b�D�	w:K���A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) �܀d:>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	   �K�(@   �K�(@      �?!   �K�(@) "ZldBc@2e��:�(@����t*@�������:              �?        ���       b�D�	��QL���A�*�
w
discriminator_loss*a	    R��>    R��>      �?!    R��>)  �e0�>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   @[8)@   @[8)@      �?!   @[8)@)�4d�q�c@2e��:�(@����t*@�������:              �?        ��x��       b�D�	�w�M���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) ­�(�G>2��ڋ?�.�?�������:              �?        
s
generator_loss*a	   @t{)@   @t{)@      �?!   @t{)@)�PR��Jd@2e��:�(@����t*@�������:              �?        �}	�       �N�	���N���A*�
w
discriminator_loss*a	   `�?   `�?      �?!   `�?)@���1>2x?�x�?��d�r?�������:              �?        
s
generator_loss*a	   @8�)@   @8�)@      �?!   @8�)@)�஡�d@2e��:�(@����t*@�������:              �?        �5���       �{�	��O���A(*�
w
discriminator_loss*a	    �:?    �:?      �?!    �:?)  A�p*>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	   ���)@   ���)@      �?!   ���)@) ս�e@2e��:�(@����t*@�������:              �?        _g��       �{�	�BQ���AP*�
w
discriminator_loss*a	   �/� ?   �/� ?      �?!   �/� ?)@@S(a#>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   ��*@   ��*@      �?!   ��*@) �'��*e@2e��:�(@����t*@�������:              �?        �ڥ$�       �{�	qP�R���Ax*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)�<4i}�!>2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	    m)@    m)@      �?!    m)@) ����3d@2e��:�(@����t*@�������:              �?        �_���       b�D�	/��S���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �ψ_�=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   �u*@   �u*@      �?!   �u*@)�H�-�?e@2e��:�(@����t*@�������:              �?        ��`=�       b�D�	)�U���A�*�
w
discriminator_loss*a	   �Y
�>   �Y
�>      �?!   �Y
�>) �dk3�=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   @io*@   @io*@      �?!   @io*@)�,���e@2e��:�(@����t*@�������:              �?        ��TM�       b�D�	cgV���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@�뙍>2��[�?1��a˲?�������:              �?        
s
generator_loss*a	   @z*@   @z*@      �?!   @z*@)��˽#�e@2����t*@�}h�-@�������:              �?        ���"�       b�D�	��W���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) 9 v-/>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   @��*@   @��*@      �?!   @��*@)�ܪ��'f@2����t*@�}h�-@�������:              �?        ���)�       b�D�	�Y���A�*�
w
discriminator_loss*a	   �	*�>   �	*�>      �?!   �	*�>)@ʍ����=2�f����>��(���>�������:              �?        
s
generator_loss*a	   ���*@   ���*@      �?!   ���*@)��`��f@2����t*@�}h�-@�������:              �?        �� �       b�D�	��UZ���A�*�
w
discriminator_loss*a	    t%�>    t%�>      �?!    t%�>)@�^W��=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   @�+@   @�+@      �?!   @�+@)�TV�q�f@2����t*@�}h�-@�������:              �?        ܫ�       b�D�	 s�[���A�*�
w
discriminator_loss*a	   �I�	?   �I�	?      �?!   �I�	?) �p�m�$>2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	   `�"+@   `�"+@      �?!   `�"+@) ��]g@2����t*@�}h�-@�������:              �?        ����       b�D�	�d�\���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) 	��>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	   ��/+@   ��/+@      �?!   ��/+@) �b�{g@2����t*@�}h�-@�������:              �?        ��$w�       b�D�	�_P^���A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)�,��C�>2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   ��Z+@   ��Z+@      �?!   ��Z+@)�8�bg@2����t*@�}h�-@�������:              �?        ���J�       b�D�	���_���A�*�
w
discriminator_loss*a	    7��>    7��>      �?!    7��>) �N���=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   �hq+@   �hq+@      �?!   �hq+@)��"��g@2����t*@�}h�-@�������:              �?        ���Y�       b�D�	P�`���A�*�
w
discriminator_loss*a	    $�>    $�>      �?!    $�>) I�	N>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   ��+@   ��+@      �?!   ��+@) ��!h@2����t*@�}h�-@�������:              �?        ��       b�D�	-Zb���A�*�
w
discriminator_loss*a	   �hq�>   �hq�>      �?!   �hq�>) B]�H�=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	    ��+@    ��+@      �?!    ��+@) ����g@2����t*@�}h�-@�������:              �?        ��R�       b�D�	DF�c���A�*�
w
discriminator_loss*a	   �Ƕ�>   �Ƕ�>      �?!   �Ƕ�>)��!��>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   ���+@   ���+@      �?!   ���+@) ��_U�g@2����t*@�}h�-@�������:              �?        \I�5�       b�D�	�ie���A�*�
w
discriminator_loss*a	   �*�
?   �*�
?      �?!   �*�
?) r����&>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   ���+@   ���+@      �?!   ���+@) ��f)�g@2����t*@�}h�-@�������:              �?        ���v�       b�D�	0�sf���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ��V�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   ��+@   ��+@      �?!   ��+@) 2�nJh@2����t*@�}h�-@�������:              �?        }|(�       b�D�	���g���A�*�
w
discriminator_loss*a	   ��U�>   ��U�>      �?!   ��U�>)��[����=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   �,@   �,@      �?!   �,@) �����h@2����t*@�}h�-@�������:              �?        ӎkP�       b�D�	'�6i���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@�1̎��=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   �IO,@   �IO,@      �?!   �IO,@) Ґ�i@2����t*@�}h�-@�������:              �?        \|��       b�D�	3�j���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) hĽ*�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   @;L,@   @;L,@      �?!   @;L,@)���Ii@2����t*@�}h�-@�������:              �?        w����       �N�	"��k���A*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ђ�/�=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   @�M,@   @�M,@      �?!   @�M,@)�ݑ'	i@2����t*@�}h�-@�������:              �?        ����       �{�	(Um���A(*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@�ɰ{9�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   �CN,@   �CN,@      �?!   �CN,@)�p���	i@2����t*@�}h�-@�������:              �?        �����       �{�	��n���AP*�
w
discriminator_loss*a	    �
�>    �
�>      �?!    �
�>)  ��d�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   ��T,@   ��T,@      �?!   ��T,@) ����i@2����t*@�}h�-@�������:              �?        ��}��       �{�	��3p���Ax*�
w
discriminator_loss*a	   ��l�>   ��l�>      �?!   ��l�>) r�uҤ�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   @0s,@   @0s,@      �?!   @0s,@)��3Ki@2����t*@�}h�-@�������:              �?        �tg�       b�D�	f��q���A�*�
w
discriminator_loss*a	   @�k�>   @�k�>      �?!   @�k�>)��(��=�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   ��,@   ��,@      �?!   ��,@)�0 +��i@2����t*@�}h�-@�������:              �?        �2ێ�       b�D�	3Gs���A�*�
w
discriminator_loss*a	   �~a�>   �~a�>      �?!   �~a�>)�b�m�=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   �$�,@   �$�,@      �?!   �$�,@) ���j@2����t*@�}h�-@�������:              �?        �)3�       b�D�	/�t���A�*�
w
discriminator_loss*a	    i�>    i�>      �?!    i�>) Z�I�O�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   �"�,@   �"�,@      �?!   �"�,@) 2-��%j@2����t*@�}h�-@�������:              �?        ��)��       b�D�	��u���A�*�
w
discriminator_loss*a	    0��>    0��>      �?!    0��>)  H�6E�=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   �,-@   �,-@      �?!   �,-@) �e�_j@2����t*@�}h�-@�������:              �?        	���       b�D�	�lw���A�*�
w
discriminator_loss*a	    *��>    *��>      �?!    *��>) @.(�B�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   ��-@   ��-@      �?!   ��-@) ���ij@2����t*@�}h�-@�������:              �?        ?5���       b�D�	��x���A�*�
w
discriminator_loss*a	   �tK�>   �tK�>      �?!   �tK�>) "�[,�>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	    �'-@    �'-@      �?!    �'-@) ��~�j@2�}h�-@�x�a0@�������:              �?        �k�       b�D�	H�Tz���A�*�
w
discriminator_loss*a	   @mq�>   @mq�>      �?!   @mq�>) ��{�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   �6-@   �6-@      �?!   �6-@) r�`b�j@2�}h�-@�x�a0@�������:              �?        �N��       b�D�	;��{���A�*�
w
discriminator_loss*a	    l�>    l�>      �?!    l�>) @�l�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   �J-@   �J-@      �?!   �J-@) BLc��j@2�}h�-@�x�a0@�������:              �?        ��X+�       b�D�	�7}���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@���	�=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   @�q-@   @�q-@      �?!   @�q-@)�4R~�k@2�}h�-@�x�a0@�������:              �?        �(���       b�D�	
��~���A�*�
w
discriminator_loss*a	   @L;�>   @L;�>      �?!   @L;�>) a�۰�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	    ��-@    ��-@      �?!    ��-@)  �Y#Ik@2�}h�-@�x�a0@�������:              �?        ��p��       b�D�	%8#����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@��.�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   @M�-@   @M�-@      �?!   @M�-@)�|�2cqk@2�}h�-@�x�a0@�������:              �?        ����       b�D�	�6�����A�*�
w
discriminator_loss*a	   �#��>   �#��>      �?!   �#��>) b��fx�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	    +�-@    +�-@      �?!    +�-@) ����k@2�}h�-@�x�a0@�������:              �?        ���x�       b�D�	1�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) 3��m��=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	    ��-@    ��-@      �?!    ��-@)  ��v�k@2�}h�-@�x�a0@�������:              �?        � >��       b�D�	ю����A�*�
w
discriminator_loss*a	   �n��>   �n��>      �?!   �n��>)@T@���=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   ���-@   ���-@      �?!   ���-@)�T,��Bk@2�}h�-@�x�a0@�������:              �?        ����       b�D�	0g����A�*�
w
discriminator_loss*a	    �b�>    �b�>      �?!    �b�>)@3��ǰ=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   �ʪ-@   �ʪ-@      �?!   �ʪ-@) r���k@2�}h�-@�x�a0@�������:              �?        u/0^�       b�D�	�K�����A�*�
w
discriminator_loss*a	   �M��>   �M��>      �?!   �M��>)@�|.D�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   �'�-@   �'�-@      �?!   �'�-@)�`I���k@2�}h�-@�x�a0@�������:              �?        �����       b�D�	99����A�*�
w
discriminator_loss*a	   �X��>   �X��>      �?!   �X��>) sM���=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   � �-@   � �-@      �?!   � �-@) �\�k@2�}h�-@�x�a0@�������:              �?        1�Iq�       b�D�	�������A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�|U��	�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   �ͽ-@   �ͽ-@      �?!   �ͽ-@)�薷j�k@2�}h�-@�x�a0@�������:              �?        ��3F�       �N�	 ������A*�
w
discriminator_loss*a	   ��Q?   ��Q?      �?!   ��Q?)��2���,>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	    T�-@    T�-@      �?!    T�-@) �@h~k@2�}h�-@�x�a0@�������:              �?        .�!�       �{�	�������A(*�
w
discriminator_loss*a	    t��>    t��>      �?!    t��>)@�VnM�=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   ��-@   ��-@      �?!   ��-@) ��k@2�}h�-@�x�a0@�������:              �?        ����       �{�	}����AP*�
w
discriminator_loss*a	   ��K�>   ��K�>      �?!   ��K�>) ����=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   `W�-@   `W�-@      �?!   `W�-@) ��(��k@2�}h�-@�x�a0@�������:              �?        tCnQ�       �{�	�Q�����Ax*�
w
discriminator_loss*a	   ��'�>   ��'�>      �?!   ��'�>)@�Yxd�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   @d.@   @d.@      �?!   @d.@)�T\&l@2�}h�-@�x�a0@�������:              �?        �$�r�       b�D�	.�#����A�*�
w
discriminator_loss*a	   �h�>   �h�>      �?!   �h�>) �mW8,�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   @��-@   @��-@      �?!   @��-@)�8�bf�k@2�}h�-@�x�a0@�������:              �?        ���[�       b�D�	������A�*�
w
discriminator_loss*a	   �'��>   �'��>      �?!   �'��>) g++�=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	    V.@    V.@      �?!    V.@)  ��vQl@2�}h�-@�x�a0@�������:              �?        �)��       b�D�	n�?����A�*�
w
discriminator_loss*a	    ;��>    ;��>      �?!    ;��>) ���{>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   @�=.@   @�=.@      �?!   @�=.@)��[�l@2�}h�-@�x�a0@�������:              �?        J���       b�D�	�ϖ���A�*�
w
discriminator_loss*a	   �M�>   �M�>      �?!   �M�>)�l���=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   `�Z.@   `�Z.@      �?!   `�Z.@) Q�1�l@2�}h�-@�x�a0@�������:              �?        @��o�       b�D�	?�a����A�*�
w
discriminator_loss*a	   @�
�>   @�
�>      �?!   @�
�>)��&o��=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �@�.@   �@�.@      �?!   �@�.@) ���{m@2�}h�-@�x�a0@�������:              �?        i¸��       b�D�	�������A�*�
w
discriminator_loss*a	   @�̰>   @�̰>      �?!   @�̰>) �Uq�q=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   ��s.@   ��s.@      �?!   ��s.@)����x�l@2�}h�-@�x�a0@�������:              �?        ��c�       b�D�	8�����A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ���sW�=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   @�q.@   @�q.@      �?!   @�q.@)����C�l@2�}h�-@�x�a0@�������:              �?        9�~M�       b�D�	�� ����A�*�
w
discriminator_loss*a	   �.��>   �.��>      �?!   �.��>) �D�DB�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �Ƃ.@   �Ƃ.@      �?!   �Ƃ.@) ߤ�Jm@2�}h�-@�x�a0@�������:              �?        �%���       b�D�	W㸞���A�*�
w
discriminator_loss*a	   ��]�>   ��]�>      �?!   ��]�>)@B*<��=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   �
�.@   �
�.@      �?!   �
�.@)��7 �\m@2�}h�-@�x�a0@�������:              �?        ?����       b�D�	c�S����A�*�
w
discriminator_loss*a	   �`��>   �`��>      �?!   �`��>) F��`[�=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   ���.@   ���.@      �?!   ���.@) ��U�m@2�}h�-@�x�a0@�������:              �?        �$Z1�       b�D�	n�����A�*�
w
discriminator_loss*a	   ��v?   ��v?      �?!   ��v?)@o{�N>2��[�?1��a˲?�������:              �?        
s
generator_loss*a	   ���.@   ���.@      �?!   ���.@)���GT�m@2�}h�-@�x�a0@�������:              �?        R3���       b�D�	M������A�*�
w
discriminator_loss*a	    H �>    H �>      �?!    H �>)  D��=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   �K�.@   �K�.@      �?!   �K�.@) ���dm@2�}h�-@�x�a0@�������:              �?        �����       b�D�	��4����A�*�
w
discriminator_loss*a	   `^�>   `^�>      �?!   `^�>) U>�2N�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   `¼.@   `¼.@      �?!   `¼.@) ���E�m@2�}h�-@�x�a0@�������:              �?        �$pM�       b�D�	�Ӧ���A�*�
w
discriminator_loss*a	    �#�>    �#�>      �?!    �#�>) �,�߿�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   @H�.@   @H�.@      �?!   @H�.@)� ��u�m@2�}h�-@�x�a0@�������:              �?        ���       b�D�	�Kv����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) D���N�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   @��.@   @��.@      �?!   @��.@)�0(�m@2�}h�-@�x�a0@�������:              �?        �����       b�D�	������A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �g�g�=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   �W�.@   �W�.@      �?!   �W�.@) Bw��n@2�}h�-@�x�a0@�������:              �?        yT�-�       b�D�	\������A�*�
w
discriminator_loss*a	   ��Z�>   ��Z�>      �?!   ��Z�>)@861��=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   �!/@   �!/@      �?!   �!/@) �:)n@2�}h�-@�x�a0@�������:              �?        C�]��       b�D�	L�i����A�*�
w
discriminator_loss*a	   `�(�>   `�(�>      �?!   `�(�>) ?c&:Wn=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   `�*/@   `�*/@      �?!   `�*/@) ��_[n@2�}h�-@�x�a0@�������:              �?        ��vx�       �N�	������A*�
w
discriminator_loss*a	   �R��>   �R��>      �?!   �R��>) ��~h��=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	    �:/@    �:/@      �?!    �:/@)  r�zn@2�}h�-@�x�a0@�������:              �?        �#��       �{�	�ܤ����A(*�
w
discriminator_loss*a	   ��0�>   ��0�>      �?!   ��0�>) [�#�΀=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   @�)/@   @�)/@      �?!   @�)/@)�(_�Yn@2�}h�-@�x�a0@�������:              �?        .x�	�       �{�	�TQ����AP*�
w
discriminator_loss*a	    U�>    U�>      �?!    U�>) ����=2�f����>��(���>�������:              �?        
s
generator_loss*a	   �l/@   �l/@      �?!   �l/@)��%�En@2�}h�-@�x�a0@�������:              �?        %��       �{�	�>�����Ax*�
w
discriminator_loss*a	   `�P�>   `�P�>      �?!   `�P�>)@���輲=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   �/@   �/@      �?!   �/@) 	�JbBn@2�}h�-@�x�a0@�������:              �?        ��9��       b�D�	n
�����A�*�
w
discriminator_loss*a	    �o�>    �o�>      �?!    �o�>)@�w?/��=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   `A(/@   `A(/@      �?!   `A(/@) ��M1Vn@2�}h�-@�x�a0@�������:              �?        ����       b�D�	��`����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �x��=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   ��[/@   ��[/@      �?!   ��[/@) �.��n@2�}h�-@�x�a0@�������:              �?        4/���       b�D�	L�����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@8�}��=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   ��S/@   ��S/@      �?!   ��S/@) r.gr�n@2�}h�-@�x�a0@�������:              �?        �TN�       b�D�	?̺���A�*�
w
discriminator_loss*a	    �>�>    �>�>      �?!    �>�>) �t���=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   ��c/@   ��c/@      �?!   ��c/@)�(y�v�n@2�}h�-@�x�a0@�������:              �?        �٥'�       b�D�	J������A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) 2����f=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   �~/@   �~/@      �?!   �~/@) 2k0J�n@2�}h�-@�x�a0@�������:              �?        �%�m�       b�D�	T�7����A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) ����s=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   @�/@   @�/@      �?!   @�/@)����]?o@2�}h�-@�x�a0@�������:              �?        I0gT�       b�D�	S�����A�*�
w
discriminator_loss*a	   �A+�>   �A+�>      �?!   �A+�>) ��hƀ=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   `g�/@   `g�/@      �?!   `g�/@) �PE�o@2�}h�-@�x�a0@�������:              �?        �eߐ�       b�D�	t�����A�*�
w
discriminator_loss*a	   �$��>   �$��>      �?!   �$��>) i�L=>q=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   �A�/@   �A�/@      �?!   �A�/@)���Ӛo@2�}h�-@�x�a0@�������:              �?        +|���       b�D�	������A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@n0x/�=2�f����>��(���>�������:              �?        
s
generator_loss*a	   �g�/@   �g�/@      �?!   �g�/@) ��
bwo@2�}h�-@�x�a0@�������:              �?        ��Y�       b�D�	�MJ����A�*�
w
discriminator_loss*a	   @�ٵ>   @�ٵ>      �?!   @�ٵ>) !�?��}=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   ���/@   ���/@      �?!   ���/@) ��Av�o@2�}h�-@�x�a0@�������:              �?        ���       b�D�	(�����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@���v=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	    �/@    �/@      �?!    �/@) ���3�o@2�}h�-@�x�a0@�������:              �?        +�>�       b�D�	������A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �N}�<�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   @�0@   @�0@      �?!   @�0@) �P�	p@2�x�a0@�����1@�������:              �?        �B��       b�D�	�J�����A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)�P(��sh=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   �m
0@   �m
0@      �?!   �m
0@) � L�p@2�x�a0@�����1@�������:              �?        /wt�       b�D�	K�X����A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) @�^�hu=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	    u0@    u0@      �?!    u0@) �w�+p@2�x�a0@�����1@�������:              �?        ���       b�D�	������A�*�
w
discriminator_loss*a	   �*�>   �*�>      �?!   �*�>) �`@ G�=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   �M0@   �M0@      �?!   �M0@) �!ʲ&p@2�x�a0@�����1@�������:              �?        k}�6�       b�D�	������A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�����.`=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	    50@    50@      �?!    50@) �o�<p@2�x�a0@�����1@�������:              �?        W����       b�D�	�ڵ����A�*�
w
discriminator_loss*a	   `g��>   `g��>      �?!   `g��>)@�{���=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   �u+0@   �u+0@      �?!   �u+0@) ��
aWp@2�x�a0@�����1@�������:              �?        �S9��       b�D�	 ������A�*�
w
discriminator_loss*a	   `5b�>   `5b�>      �?!   `5b�>)@�>��y=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   `<80@   `<80@      �?!   `<80@)@҃g>qp@2�x�a0@�����1@�������:              �?        �{��       �N�	s�5����A*�
w
discriminator_loss*a	   �@-�>   �@-�>      �?!   �@-�>)@�x��6=2�4[_>��>
�}���>�������:              �?        
s
generator_loss*a	    N;0@    N;0@      �?!    N;0@) @��wwp@2�x�a0@�����1@�������:              �?        O'���       �{�	�����A(*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)���&��=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   @�;0@   @�;0@      �?!   @�;0@) Q���wp@2�x�a0@�����1@�������:              �?        �M)c�       �{�	�j�����AP*�
w
discriminator_loss*a	    �ק>    �ק>      �?!    �ק>) �����a=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	    �<0@    �<0@      �?!    �<0@)@����zp@2�x�a0@�����1@�������:              �?        �����       �{�	�@�����Ax*�
w
discriminator_loss*a	   `�,�>   `�,�>      �?!   `�,�>) i�)�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   �?M0@   �?M0@      �?!   �?M0@)@ ˷��p@2�x�a0@�����1@�������:              �?        ����       b�D�	z`�����A�*�
w
discriminator_loss*a	   `W�>   `W�>      �?!   `W�>)@&��:�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   ��U0@   ��U0@      �?!   ��U0@) yŅ��p@2�x�a0@�����1@�������:              �?        Ҕ,-�       b�D�	�>^����A�*�
w
discriminator_loss*a	   @nؖ>   @nؖ>      �?!   @nؖ>)���iOO@=2X$�z�>.��fc��>�������:              �?        
s
generator_loss*a	   �-\0@   �-\0@      �?!   �-\0@)@2Mn�p@2�x�a0@�����1@�������:              �?        ��`�       b�D�	�D6����A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) hj�=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   `�]0@   `�]0@      �?!   `�]0@)@Z�,��p@2�x�a0@�����1@�������:              �?        ,���       b�D�	5����A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)���3 >2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   ��I0@   ��I0@      �?!   ��I0@) ���$�p@2�x�a0@�����1@�������:              �?        ���?�       b�D�	F�����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  �w^=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   �iS0@   �iS0@      �?!   �iS0@) �Gم�p@2�x�a0@�����1@�������:              �?        �t� �       b�D�	������A�*�
w
discriminator_loss*a	   @>   @>      �?!   @>)���(�I=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	   �`V0@   �`V0@      �?!   �`V0@)@�����p@2�x�a0@�����1@�������:              �?        ÅA��       b�D�	�ȧ����A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) ��ġ�w=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   `Y0@   `Y0@      �?!   `Y0@)@���p@2�x�a0@�����1@�������:              �?        ��E�       b�D�	�������A�*�
w
discriminator_loss*a	    .>�>    .>�>      �?!    .>�>)@��1�$W=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   @Te0@   @Te0@      �?!   @Te0@) �c9*�p@2�x�a0@�����1@�������:              �?        =GL�       b�D�	�bk����A�*�
w
discriminator_loss*a	   ��ֵ>   ��ֵ>      �?!   ��ֵ>)@nw��}=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   �el0@   �el0@      �?!   �el0@) '��p@2�x�a0@�����1@�������:              �?        ��-�       b�D�	�L����A�*�
w
discriminator_loss*a	   �ᔠ>   �ᔠ>      �?!   �ᔠ>)@�$-/Q=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	   �w0@   �w0@      �?!   �w0@)@XS���p@2�x�a0@�����1@�������:              �?        h�{��       b�D�	��2����A�*�
w
discriminator_loss*a	   �X[�>   �X[�>      �?!   �X[�>)�$��*c�=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   `�~0@   `�~0@      �?!   `�~0@)@�,Pq@2�x�a0@�����1@�������:              �?        �T�:�       b�D�	 ����A�*�
w
discriminator_loss*a	    �z�>    �z�>      �?!    �z�>) b���=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   �S�0@   �S�0@      �?!   �S�0@) ��Rq@2�x�a0@�����1@�������:              �?        Rh�0�       b�D�	������A�*�
w
discriminator_loss*a	   `k�>   `k�>      �?!   `k�>)@��J=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   ���0@   ���0@      �?!   ���0@) �gK�2q@2�x�a0@�����1@�������:              �?        S�Ύ�       b�D�	2�����A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) �kA��=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	    ��0@    ��0@      �?!    ��0@)@teӃ,q@2�x�a0@�����1@�������:              �?        ���       b�D�	 ������A�*�
w
discriminator_loss*a	    �V�>    �V�>      �?!    �V�>)@�S�0`W=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	    �0@    �0@      �?!    �0@) ���5#q@2�x�a0@�����1@�������:              �?        �b�`�       b�D�	�����A�*�
w
discriminator_loss*a	   @7��>   @7��>      �?!   @7��>)�d;��M�=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   �&t0@   �&t0@      �?!   �&t0@)@t���p@2�x�a0@�����1@�������:              �?        ����       b�D�	5>�����A�*�
w
discriminator_loss*a	   �hX�>   �hX�>      �?!   �hX�>)��v�
A=2X$�z�>.��fc��>�������:              �?        
s
generator_loss*a	   @��0@   @��0@      �?!   @��0@) q�m�q@2�x�a0@�����1@�������:              �?        ��       b�D�	������A�*�
w
discriminator_loss*a	   @j��>   @j��>      �?!   @j��>) �	}��u=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   ���0@   ���0@      �?!   ���0@) �/�q@2�x�a0@�����1@�������:              �?        �@�	�       �N�	�C�����A*�
w
discriminator_loss*a	   @5i�>   @5i�>      �?!   @5i�>)���q� A=2X$�z�>.��fc��>�������:              �?        
s
generator_loss*a	   ���0@   ���0@      �?!   ���0@) ���](q@2�x�a0@�����1@�������:              �?        K�CZ�       �{�	��| ���A(*�
w
discriminator_loss*a	   ��۠>   ��۠>      �?!   ��۠>) �O=��Q=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	    �0@    �0@      �?!    �0@)@��yWq@2�x�a0@�����1@�������:              �?        [+�       �{�	�t���AP*�
w
discriminator_loss*a	   @	�>   @	�>      �?!   @	�>)��2T�zj=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   @�0@   @�0@      �?!   @�0@) ��	xq@2�x�a0@�����1@�������:              �?        ���       �{�	6-j���Ax*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�,�oD��=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   @��0@   @��0@      �?!   @��0@) q(�ώq@2�x�a0@�����1@�������:              �?        �n��       b�D�	�f���A�*�
w
discriminator_loss*a	   `c�>   `c�>      �?!   `c�>) ���=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   �ɸ0@   �ɸ0@      �?!   �ɸ0@) �i$�yq@2�x�a0@�����1@�������:              �?        ̘�n�       b�D�	T�^���A�*�
w
discriminator_loss*a	   @m�>   @m�>      �?!   @m�>) ��t�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   ��0@   ��0@      �?!   ��0@) ��\aq@2�x�a0@�����1@�������:              �?        *-X-�       b�D�	 �]
���A�*�
w
discriminator_loss*a	   @  �>   @  �>      �?!   @  �>)  �@@=2�����~>[#=�؏�>�������:              �?        
s
generator_loss*a	   ��0@   ��0@      �?!   ��0@) $?�gq@2�x�a0@�����1@�������:              �?        ��Q�       b�D�	Y
\���A�*�
w
discriminator_loss*a	   �ou�>   �ou�>      �?!   �ou�>) ���(:=2
�}���>X$�z�>�������:              �?        
s
generator_loss*a	    ��0@    ��0@      �?!    ��0@)@��͒{q@2�x�a0@�����1@�������:              �?        $m�x�       b�D�	ehZ���A�*�
w
discriminator_loss*a	   @s��>   @s��>      �?!   @s��>) )>�A��=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   �W�0@   �W�0@      �?!   �W�0@) A�?�q@2�x�a0@�����1@�������:              �?        ?ʿ��       b�D�	#�Z���A�*�
w
discriminator_loss*a	    �}�>    �}�>      �?!    �}�>) �x�>�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	    ��0@    ��0@      �?!    ��0@) ��˰q@2�x�a0@�����1@�������:              �?        *�7�       b�D�	�7b���A�*�
w
discriminator_loss*a	   �t��>   �t��>      �?!   �t��>) ���B=2.��fc��>39W$:��>�������:              �?        
s
generator_loss*a	   ���0@   ���0@      �?!   ���0@)@��j?�q@2�x�a0@�����1@�������:              �?        ���       b�D�	�Af���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �5B�9=2
�}���>X$�z�>�������:              �?        
s
generator_loss*a	   �%�0@   �%�0@      �?!   �%�0@) �?��q@2�x�a0@�����1@�������:              �?        H�)�       b�D�	Qm���A�*�
w
discriminator_loss*a	   ��f�>   ��f�>      �?!   ��f�>) �1%&�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   ���0@   ���0@      �?!   ���0@) ���I�q@2�x�a0@�����1@�������:              �?        �t4��       b�D�	��w���A�*�
w
discriminator_loss*a	   �	B�>   �	B�>      �?!   �	B�>)��:Lv�@=2X$�z�>.��fc��>�������:              �?        
s
generator_loss*a	   ���0@   ���0@      �?!   ���0@) D�ц�q@2�x�a0@�����1@�������:              �?        `Ǥ��       b�D�	�����A�*�
w
discriminator_loss*a	    �b�>    �b�>      �?!    �b�>)  �d�=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   ���0@   ���0@      �?!   ���0@)@�=��q@2�x�a0@�����1@�������:              �?        e�$�       b�D�	������A�*�
w
discriminator_loss*a	    e�>    e�>      �?!    e�>) �8�L=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	    ��0@    ��0@      �?!    ��0@)@ǯ\�q@2�x�a0@�����1@�������:              �?        �w�
�       b�D�	�Þ���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) Y���>�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   �0�0@   �0�0@      �?!   �0�0@) �N��q@2�x�a0@�����1@�������:              �?         ?q�       b�D�	��� ���A�*�
w
discriminator_loss*a	    4��>    4��>      �?!    4��>) �T�!�B=2.��fc��>39W$:��>�������:              �?        
s
generator_loss*a	   ���0@   ���0@      �?!   ���0@) q��h�q@2�x�a0@�����1@�������:              �?        Nd�k�       b�D�	���"���A�*�
w
discriminator_loss*a	   �U��>   �U��>      �?!   �U��>)@�蓁S=2u��6
�>T�L<�>�������:              �?        
s
generator_loss*a	   @�1@   @�1@      �?!   @�1@) ���r@2�x�a0@�����1@�������:              �?        �ӊv�       b�D�	��$���A�*�
w
discriminator_loss*a	    �#�>    �#�>      �?!    �#�>) �F�پh=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   ��1@   ��1@      �?!   ��1@) �DʇDr@2�x�a0@�����1@�������:              �?        ����       b�D�	;�&���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) yF��^T=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   ��'1@   ��'1@      �?!   ��'1@) D�,jdr@2�x�a0@�����1@�������:              �?         ����       b�D�	,�)���A�*�
w
discriminator_loss*a	   ��T�>   ��T�>      �?!   ��T�>) d�+��R=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   ��&1@   ��&1@      �?!   ��&1@) Y�:+cr@2�x�a0@�����1@�������:              �?        ��9��       �N�	n�+���A*�
w
discriminator_loss*a	   `�߆>   `�߆>      �?!   `�߆>) ۲S�Y =2T�L<�>��z!�?�>�������:              �?        
s
generator_loss*a	    �-1@    �-1@      �?!    �-1@) R�rr@2�x�a0@�����1@�������:              �?        ��8��       �{�	��--���A(*�
w
discriminator_loss*a	   @�>�>   @�>�>      �?!   @�>�>) q�� ~p=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   ��61@   ��61@      �?!   ��61@) ���r@2�x�a0@�����1@�������:              �?        ��6�       �{�	�K/���AP*�
w
discriminator_loss*a	   �ߩ�>   �ߩ�>      �?!   �ߩ�>)@�I���z=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   �PA1@   �PA1@      �?!   �PA1@) ��9֛r@2�x�a0@�����1@�������:              �?        Mb�p�       �{�	��j1���Ax*�
w
discriminator_loss*a	   `4^�>   `4^�>      �?!   `4^�>)@r�:E=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	    �?1@    �?1@      �?!    �?1@) �sH��r@2�x�a0@�����1@�������:              �?        0�$g�       b�D�	a؆3���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) Aݻ)ژ=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   ��A1@   ��A1@      �?!   ��A1@)@��o�r@2�x�a0@�����1@�������:              �?        �����       b�D�	:��5���A�*�
w
discriminator_loss*a	   @KP�>   @KP�>      �?!   @KP�>)����Q��=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   `�?1@   `�?1@      �?!   `�?1@)@.5��r@2�x�a0@�����1@�������:              �?        � ���       b�D�	Z��7���A�*�
w
discriminator_loss*a	   ��?>   ��?>      �?!   ��?>)�h
[\�=2�����~>[#=�؏�>�������:              �?        
s
generator_loss*a	    %11@    %11@      �?!    %11@) ���yr@2�x�a0@�����1@�������:              �?        ׭lU�       b�D�	%s�9���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)���Rx`=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   ��/1@   ��/1@      �?!   ��/1@)@FǺ�ur@2�x�a0@�����1@�������:              �?        D%�,�       b�D�	�<���A�*�
w
discriminator_loss*a	    š>    š>      �?!    š>)@�
D�S=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	    �81@    �81@      �?!    �81@) @4C��r@2�x�a0@�����1@�������:              �?        �h�       b�D�	�1>>���A�*�
w
discriminator_loss*a	   �*ښ>   �*ښ>      �?!   �*ښ>) reU�F=239W$:��>R%�����>�������:              �?        
s
generator_loss*a	    �M1@    �M1@      �?!    �M1@)@�v�l�r@2�x�a0@�����1@�������:              �?        o4���       b�D�	Li@���A�*�
w
discriminator_loss*a	    �Z�>    �Z�>      �?!    �Z�>)  ��d=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	    �K1@    �K1@      �?!    �K1@) ��7Ѳr@2�x�a0@�����1@�������:              �?        �欵�       b�D�	(y�B���A�*�
w
discriminator_loss*a	    E��>    E��>      �?!    E��>) ����\�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   `�T1@   `�T1@      �?!   `�T1@)@�16;�r@2�x�a0@�����1@�������:              �?        q���       b�D�	v��D���A�*�
w
discriminator_loss*a	   ൺ�>   ൺ�>      �?!   ൺ�>)@h�|I�z=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   ��`1@   ��`1@      �?!   ��`1@)@�fO3�r@2�x�a0@�����1@�������:              �?        ����       b�D�	e��F���A�*�
w
discriminator_loss*a	   �5��>   �5��>      �?!   �5��>)@h=x��5=2�4[_>��>
�}���>�������:              �?        
s
generator_loss*a	   ��j1@   ��j1@      �?!   ��j1@) 9y���r@2�x�a0@�����1@�������:              �?        �O"�       b�D�	�I���A�*�
w
discriminator_loss*a	    yWr>    yWr>      �?!    yWr>) sw��<2ہkVl�p>BvŐ�r>�������:              �?        
s
generator_loss*a	   �*{1@   �*{1@      �?!   �*{1@) 9��ns@2�x�a0@�����1@�������:              �?        E{��       b�D�	��IK���A�*�
w
discriminator_loss*a	   �I;�>   �I;�>      �?!   �I;�>)@)&�=2K���7�>u��6
�>�������:              �?        
s
generator_loss*a	   ��1@   ��1@      �?!   ��1@) D��93s@2�x�a0@�����1@�������:              �?        �.��       b�D�	�|M���A�*�
w
discriminator_loss*a	   @s,~>   @s,~>      �?!   @s,~>)��֕s=2�����~>[#=�؏�>�������:              �?        
s
generator_loss*a	   @��1@   @��1@      �?!   @��1@) )cE!Fs@2�x�a0@�����1@�������:              �?        �\r��       b�D�	�h�O���A�*�
w
discriminator_loss*a	   �w��>   �w��>      �?!   �w��>)@bf�iu=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   �ȗ1@   �ȗ1@      �?!   �ȗ1@) ��~*Xs@2�x�a0@�����1@�������:              �?        Z�x��       b�D�	���Q���A�*�
w
discriminator_loss*a	   �;ؐ>   �;ؐ>      �?!   �;ؐ>) !��1=2���m!#�>�4[_>��>�������:              �?        
s
generator_loss*a	   `i�1@   `i�1@      �?!   `i�1@)@��	2s@2�x�a0@�����1@�������:              �?        �����       b�D�	�#T���A�*�
w
discriminator_loss*a	   `�k�>   `�k�>      �?!   `�k�>)@��t�Z=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   �O�1@   �O�1@      �?!   �O�1@) �%�)As@2�x�a0@�����1@�������:              �?        ����       b�D�	��]V���A�*�
w
discriminator_loss*a	   @+�>   @+�>      �?!   @+�>)�tn�a�-=2�
�%W�>���m!#�>�������:              �?        
s
generator_loss*a	    _�1@    _�1@      �?!    _�1@) ��uds@2�����1@q��D�]3@�������:              �?        �����       b�D�	Â�X���A�*�
w
discriminator_loss*a	   ��آ>   ��آ>      �?!   ��آ>) �/A?3V=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	    �1@    �1@      �?!    �1@)@0=Z	�s@2�����1@q��D�]3@�������:              �?        �`���       �N�	��Z���A*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@&��i`{=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	    Ϭ1@    Ϭ1@      �?!    Ϭ1@) �L��s@2�����1@q��D�]3@�������:              �?        �� �       �{�	D�\���A(*�
w
discriminator_loss*a	   @<�t>   @<�t>      �?!   @<�t>) �jmo��<2�H5�8�t>�i����v>�������:              �?        
s
generator_loss*a	     �1@     �1@      �?!     �1@)@���s@2�����1@q��D�]3@�������:              �?        �����       �{�	�V<_���AP*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �_��!=2T�L<�>��z!�?�>�������:              �?        
s
generator_loss*a	   ���1@   ���1@      �?!   ���1@) $�нs@2�����1@q��D�]3@�������:              �?        ޹�O�       �{�	���a���Ax*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) x�n��K=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	   `޺1@   `޺1@      �?!   `޺1@)@�J
��s@2�����1@q��D�]3@�������:              �?        #���       b�D�	��c���A�*�
w
discriminator_loss*a	   `l�>   `l�>      �?!   `l�>)@ޢZ=2u��6
�>T�L<�>�������:              �?        
s
generator_loss*a	   ��1@   ��1@      �?!   ��1@) D��e�s@2�����1@q��D�]3@�������:              �?        �А��       b�D�	W0f���A�*�
w
discriminator_loss*a	    s)�>    s)�>      �?!    s)�>)@\@�9�[=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   `�1@   `�1@      �?!   `�1@)@�Hiys@2�����1@q��D�]3@�������:              �?        �4���       b�D�	ɲTh���A�*�
w
discriminator_loss*a	   `@ܳ>   `@ܳ>      �?!   `@ܳ>)@S��x=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   @�1@   @�1@      �?!   @�1@) ��l�s@2�����1@q��D�]3@�������:              �?        9!�       b�D�	䄝j���A�*�
w
discriminator_loss*a	   @ޗ�>   @ޗ�>      �?!   @ޗ�>) 1�H$=2u��6
�>T�L<�>�������:              �?        
s
generator_loss*a	    8�1@    8�1@      �?!    8�1@)@��:��s@2�����1@q��D�]3@�������:              �?        �&G(�       b�D�	K��l���A�*�
w
discriminator_loss*a	    t�>    t�>      �?!    t�>) ��>���=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   @H�1@   @H�1@      �?!   @H�1@) A���s@2�����1@q��D�]3@�������:              �?        ��O�       b�D�	Ww7o���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@�M�:X=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	    ��1@    ��1@      �?!    ��1@) n$�s@2�����1@q��D�]3@�������:              �?        ����       b�D�	H�q���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��as=2K���7�>u��6
�>�������:              �?        
s
generator_loss*a	    4�1@    4�1@      �?!    4�1@)  )w��s@2�����1@q��D�]3@�������:              �?        �+�       b�D�	t��s���A�*�
w
discriminator_loss*a	   `VƖ>   `VƖ>      �?!   `VƖ>) %��5@=2X$�z�>.��fc��>�������:              �?        
s
generator_loss*a	   �շ1@   �շ1@      �?!   �շ1@)@:�%�s@2�����1@q��D�]3@�������:              �?        ^�&��       b�D�	8B,v���A�*�
w
discriminator_loss*a	    H+�>    H+�>      �?!    H+�>) �x@D'=2��ӤP��>�
�%W�>�������:              �?        
s
generator_loss*a	   @H�1@   @H�1@      �?!   @H�1@) A.���s@2�����1@q��D�]3@�������:              �?        WT���       b�D�	}x���A�*�
w
discriminator_loss*a	   �cm�>   �cm�>      �?!   �cm�>)����b=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	   ���1@   ���1@      �?!   ���1@) �q��s@2�����1@q��D�]3@�������:              �?        O����       b�D�	�@�z���A�*�
w
discriminator_loss*a	   ��c>   ��c>      �?!   ��c>) �HMh�=2�����~>[#=�؏�>�������:              �?        
s
generator_loss*a	    ��1@    ��1@      �?!    ��1@) �P�9�s@2�����1@q��D�]3@�������:              �?        ��6��       b�D�	�j(}���A�*�
w
discriminator_loss*a	   ��s>   ��s>      �?!   ��s>)@н���<2BvŐ�r>�H5�8�t>�������:              �?        
s
generator_loss*a	   ��1@   ��1@      �?!   ��1@) q��It@2�����1@q��D�]3@�������:              �?        8E�B�       b�D�	�G����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@1T�y3=2���m!#�>�4[_>��>�������:              �?        
s
generator_loss*a	     2@     2@      �?!     2@)@�X��Mt@2�����1@q��D�]3@�������:              �?        `%�h�       b�D�	�2�����A�*�
w
discriminator_loss*a	   `�0W>   `�0W>      �?!   `�0W>) �M�1��<2��x��U>Fixі�W>�������:              �?        
s
generator_loss*a	    �2@    �2@      �?!    �2@)@4�h_t@2�����1@q��D�]3@�������:              �?        w?J�       b�D�	�:����A�*�
w
discriminator_loss*a	    .8�>    .8�>      �?!    .8�>)  B�$�@=2X$�z�>.��fc��>�������:              �?        
s
generator_loss*a	    �
2@    �
2@      �?!    �
2@) @�qXt@2�����1@q��D�]3@�������:              �?        3H���       b�D�	�ؖ����A�*�
w
discriminator_loss*a	   ��,�>   ��,�>      �?!   ��,�>) �T�8�C=2.��fc��>39W$:��>�������:              �?        
s
generator_loss*a	   ��2@   ��2@      �?!   ��2@)@��]jt@2�����1@q��D�]3@�������:              �?        (�C�       b�D�	k�����A�*�
w
discriminator_loss*a	   ��?�>   ��?�>      �?!   ��?�>) ��u-(=2K���7�>u��6
�>�������:              �?        
s
generator_loss*a	    2@    2@      �?!    2@)@`���t@2�����1@q��D�]3@�������:              �?        Q��       b�D�	��V����A�*�
w
discriminator_loss*a	   ��Þ>   ��Þ>      �?!   ��Þ>) k����M=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	   �+#2@   �+#2@      �?!   �+#2@) D�.o�t@2�����1@q��D�]3@�������:              �?        ꊏ�       �N�	������A*�
w
discriminator_loss*a	   `Ѷ�>   `Ѷ�>      �?!   `Ѷ�>) ��l�A=2.��fc��>39W$:��>�������:              �?        
s
generator_loss*a	   @�!2@   @�!2@      �?!   @�!2@) ��]ߋt@2�����1@q��D�]3@�������:              �?        .5Ǐ�       �{�	�5
����A(*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)��0 ,=2�
�%W�>���m!#�>�������:              �?        
s
generator_loss*a	   `02@   `02@      �?!   `02@)@@ ��t@2�����1@q��D�]3@�������:              �?        ҥ=��       �{�	�q����AP*�
w
discriminator_loss*a	    �ؔ>    �ؔ>      �?!    �ؔ>)@��^F);=2
�}���>X$�z�>�������:              �?        
s
generator_loss*a	   �>82@   �>82@      �?!   �>82@) $tVR�t@2�����1@q��D�]3@�������:              �?        �h�       �{�	v�۔���Ax*�
w
discriminator_loss*a	   `�=x>   `�=x>      �?!   `�=x>) ��>�\=2�i����v>E'�/��x>�������:              �?        
s
generator_loss*a	   �W82@   �W82@      �?!   �W82@) A!؋�t@2�����1@q��D�]3@�������:              �?        ��=�       b�D�	+AF����A�*�
w
discriminator_loss*a	   `S�>   `S�>      �?!   `S�>)@v����=2u��6
�>T�L<�>�������:              �?        
s
generator_loss*a	   �#72@   �#72@      �?!   �#72@)@R;-μt@2�����1@q��D�]3@�������:              �?        bKU��       b�D�	�E�����A�*�
w
discriminator_loss*a	   ��o�>   ��o�>      �?!   ��o�>)��`�H�.=2�
�%W�>���m!#�>�������:              �?        
s
generator_loss*a	   `�F2@   `�F2@      �?!   `�F2@)@F���t@2�����1@q��D�]3@�������:              �?        _8e\�       b�D�	J\����A�*�
w
discriminator_loss*a	   ��x�>   ��x�>      �?!   ��x�>)���I�$+=2�
�%W�>���m!#�>�������:              �?        
s
generator_loss*a	   �IU2@   �IU2@      �?!   �IU2@)@Aڬu@2�����1@q��D�]3@�������:              �?        �?{��       b�D�	�È����A�*�
w
discriminator_loss*a	   �
�R>   �
�R>      �?!   �
�R>) �gc�<2�
L�v�Q>H��'ϱS>�������:              �?        
s
generator_loss*a	   ��Z2@   ��Z2@      �?!   ��Z2@) D=u@2�����1@q��D�]3@�������:              �?        *���       b�D�	�������A�*�
w
discriminator_loss*a	   @�L�>   @�L�>      �?!   @�L�>) �5��=2u��6
�>T�L<�>�������:              �?        
s
generator_loss*a	   �%?2@   �%?2@      �?!   �%?2@) 䧗�t@2�����1@q��D�]3@�������:              �?        7�(�       b�D�	��i����A�*�
w
discriminator_loss*a	   `Z�>   `Z�>      �?!   `Z�>)@zzn��=2K���7�>u��6
�>�������:              �?        
s
generator_loss*a	   �zG2@   �zG2@      �?!   �zG2@) ����t@2�����1@q��D�]3@�������:              �?        �sAv�       b�D�	h�ۥ���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) q]�8Q=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	   ��F2@   ��F2@      �?!   ��F2@)@�Q,�t@2�����1@q��D�]3@�������:              �?        ײ���       b�D�	��R����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) iŠ�pt=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	    �G2@    �G2@      �?!    �G2@)  Dy��t@2�����1@q��D�]3@�������:              �?        xc,��       b�D�	�ͪ���A�*�
w
discriminator_loss*a	   �&��>   �&��>      �?!   �&��>) R^�)�$=2��z!�?�>��ӤP��>�������:              �?        
s
generator_loss*a	   `�Z2@   `�Z2@      �?!   `�Z2@)@��u@2�����1@q��D�]3@�������:              �?        pi���       b�D�	ߔD����A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) V��Q=2�����~>[#=�؏�>�������:              �?        
s
generator_loss*a	   ��L2@   ��L2@      �?!   ��L2@)@�T��t@2�����1@q��D�]3@�������:              �?        I�ڨ�       b�D�	�D�����A�*�
w
discriminator_loss*a	    �p>    �p>      �?!    �p>)  	\=��<2w`f���n>ہkVl�p>�������:              �?        
s
generator_loss*a	   ��\2@   ��\2@      �?!   ��\2@) �J��u@2�����1@q��D�]3@�������:              �?        S���       b�D�	��9����A�*�
w
discriminator_loss*a	   ���p>   ���p>      �?!   ���p>) !�Ǿ�<2w`f���n>ہkVl�p>�������:              �?        
s
generator_loss*a	   �1]2@   �1]2@      �?!   �1]2@)@xg�u@2�����1@q��D�]3@�������:              �?        ^�X2�       b�D�	�����A�*�
w
discriminator_loss*a	    +�~>    +�~>      �?!    +�~>) ��߰�=2�����~>[#=�؏�>�������:              �?        
s
generator_loss*a	   @�^2@   @�^2@      �?!   @�^2@) 	`_u@2�����1@q��D�]3@�������:              �?        �V��       b�D�	��7����A�*�
w
discriminator_loss*a	   ��b~>   ��b~>      �?!   ��b~>)� �I�=2�����~>[#=�؏�>�������:              �?        
s
generator_loss*a	    _2@    _2@      �?!    _2@) @x�8u@2�����1@q��D�]3@�������:              �?        �r��       b�D�	�P¹���A�*�
w
discriminator_loss*a	   @�8d>   @�8d>      �?!   @�8d>) �꩎�<2�����0c>cR�k�e>�������:              �?        
s
generator_loss*a	   �cj2@   �cj2@      �?!   �cj2@)@p�%$2u@2�����1@q��D�]3@�������:              �?        ڐ��       b�D�	�A����A�*�
w
discriminator_loss*a	   `�0v>   `�0v>      �?!   `�0v>)@R>���<2�H5�8�t>�i����v>�������:              �?        
s
generator_loss*a	   ��q2@   ��q2@      �?!   ��q2@) aL��Bu@2�����1@q��D�]3@�������:              �?        ɠZ�       b�D�	pjȾ���A�*�
w
discriminator_loss*a	   ��7�>   ��7�>      �?!   ��7�>)@Zs�#�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	    ^�2@    ^�2@      �?!    ^�2@)@��vkpu@2�����1@q��D�]3@�������:              �?        .<�=�       b�D�	5BO����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) 9�3�=2[#=�؏�>K���7�>�������:              �?        
s
generator_loss*a	   @�|2@   @�|2@      �?!   @�|2@) yNv\u@2�����1@q��D�]3@�������:              �?        �����       �N�	�������A*�
w
discriminator_loss*a	    �ߙ>    �ߙ>      �?!    �ߙ>)  H`��D=2.��fc��>39W$:��>�������:              �?        
s
generator_loss*a	   �*�2@   �*�2@      �?!   �*�2@) 9*��ou@2�����1@q��D�]3@�������:              �?        ,�\ �       �{�	2E����A(*�
w
discriminator_loss*a	    0v�>    0v�>      �?!    0v�>) `4��PI=239W$:��>R%�����>�������:              �?        
s
generator_loss*a	    ��2@    ��2@      �?!    ��2@) @ k�u@2�����1@q��D�]3@�������:              �?        �NM��       �{�	�}�����AP*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) ��)3=2���m!#�>�4[_>��>�������:              �?        
s
generator_loss*a	   ��2@   ��2@      �?!   ��2@) ����u@2�����1@q��D�]3@�������:              �?        �L�H�       �{�	�Ok����Ax*�
w
discriminator_loss*a	   �Y؃>   �Y؃>      �?!   �Y؃>) �tB�=2K���7�>u��6
�>�������:              �?        
s
generator_loss*a	   ���2@   ���2@      �?!   ���2@)@jR��u@2�����1@q��D�]3@�������:              �?        �'�p�       b�D�	[�����A�*�
w
discriminator_loss*a	   �+��>   �+��>      �?!   �+��>)@P�8Q=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	   `��2@   `��2@      �?!   `��2@)@V�4.�u@2�����1@q��D�]3@�������:              �?        �D��       b�D�	�������A�*�
w
discriminator_loss*a	    ]��>    ]��>      �?!    ]��>)@�;>M=2[#=�؏�>K���7�>�������:              �?        
s
generator_loss*a	    �2@    �2@      �?!    �2@)  1���u@2�����1@q��D�]3@�������:              �?        e���       b�D�	�l����A�*�
w
discriminator_loss*a	    @	`>    @	`>      �?!    @	`>)@ %���<2d�V�_>w&���qa>�������:              �?        
s
generator_loss*a	   ���2@   ���2@      �?!   ���2@)@
�Ѧ�u@2�����1@q��D�]3@�������:              �?        �	��       b�D�	6�����A�*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@Ja9S�:=2
�}���>X$�z�>�������:              �?        
s
generator_loss*a	   ���2@   ���2@      �?!   ���2@) d7�<�u@2�����1@q��D�]3@�������:              �?        y�4��       b�D�	�">����A�*�
w
discriminator_loss*a	   ��*W>   ��*W>      �?!   ��*W>) ��h���<2��x��U>Fixі�W>�������:              �?        
s
generator_loss*a	   ���2@   ���2@      �?!   ���2@) �G�˿u@2�����1@q��D�]3@�������:              �?        �Vl��       b�D�	�4�����A�*�
w
discriminator_loss*a	   �J%q>   �J%q>      �?!   �J%q>) ��9�_�<2ہkVl�p>BvŐ�r>�������:              �?        
s
generator_loss*a	   �U�2@   �U�2@      �?!   �U�2@)@:v`��u@2�����1@q��D�]3@�������:              �?        �RKb�       b�D�	-�m����A�*�
w
discriminator_loss*a	    ��s>    ��s>      �?!    ��s>) ��|�<2BvŐ�r>�H5�8�t>�������:              �?        
s
generator_loss*a	   ��2@   ��2@      �?!   ��2@) D���u@2�����1@q��D�]3@�������:              �?        "�c��       b�D�	p����A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>)���_=2�����~>[#=�؏�>�������:              �?        
s
generator_loss*a	   �y�2@   �y�2@      �?!   �y�2@) qf^��u@2�����1@q��D�]3@�������:              �?        ��z�       b�D�	T������A�*�
w
discriminator_loss*a	   �"��>   �"��>      �?!   �"��>)�����Vd=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   `��2@   `��2@      �?!   `��2@)@
��v@2�����1@q��D�]3@�������:              �?        :��h�       b�D�	��C����A�*�
w
discriminator_loss*a	    Z�o>    Z�o>      �?!    Z�o>) �˷��<2w`f���n>ہkVl�p>�������:              �?        
s
generator_loss*a	   �K�2@   �K�2@      �?!   �K�2@) ����v@2�����1@q��D�]3@�������:              �?        E����       b�D�	�#�����A�*�
w
discriminator_loss*a	   �	�S>   �	�S>      �?!   �	�S>) �5~ܷ<2�
L�v�Q>H��'ϱS>�������:              �?        
s
generator_loss*a	    ��2@    ��2@      �?!    ��2@)  @ܪbv@2�����1@q��D�]3@�������:              �?        �~��       b�D�	G������A�*�
w
discriminator_loss*a	   @Xk>   @Xk>      �?!   @Xk>)�`�����<2ڿ�ɓ�i>=�.^ol>�������:              �?        
s
generator_loss*a	   `�2@   `�2@      �?!   `�2@)@��cv@2�����1@q��D�]3@�������:              �?        ��*4�       b�D�	��$����A�*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@���.q=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   �W�2@   �W�2@      �?!   �W�2@)@�cu{v@2�����1@q��D�]3@�������:              �?        �����       b�D�	{b�����A�*�
w
discriminator_loss*a	    Uup>    Uup>      �?!    Uup>)@丮��<2w`f���n>ہkVl�p>�������:              �?        
s
generator_loss*a	   ���2@   ���2@      �?!   ���2@) ���Hnv@2�����1@q��D�]3@�������:              �?        }����       b�D�	�qq����A�*�
w
discriminator_loss*a	   @sgN>   @sgN>      �?!   @sgN>)���F�<2������M>28���FP>�������:              �?        
s
generator_loss*a	    ��2@    ��2@      �?!    ��2@) @v�v@2�����1@q��D�]3@�������:              �?        ub\�       b�D�	������A�*�
w
discriminator_loss*a	   `�Q>   `�Q>      �?!   `�Q>)@����<228���FP>�
L�v�Q>�������:              �?        
s
generator_loss*a	    [�2@    [�2@      �?!    [�2@) ��#>v@2�����1@q��D�]3@�������:              �?        &Z�,�       b�D�	������A�*�
w
discriminator_loss*a	   ��;�>   ��;�>      �?!   ��;�>) ���=-'=2��ӤP��>�
�%W�>�������:              �?        
s
generator_loss*a	   @J�2@   @J�2@      �?!   @J�2@) �xU`Nv@2�����1@q��D�]3@�������:              �?        L'�       b�D�	�o����A�*�
w
discriminator_loss*a	    ̯z>    ̯z>      �?!    ̯z>) �TDqA=2E'�/��x>f^��`{>�������:              �?        
s
generator_loss*a	   ���2@   ���2@      �?!   ���2@)@r�tkuv@2�����1@q��D�]3@�������:              �?        ��g:�       �N�	 � ����A*�
w
discriminator_loss*a	   �?}>   �?}>      �?!   �?}>)� S�g\
=2f^��`{>�����~>�������:              �?        
s
generator_loss*a	    ��2@    ��2@      �?!    ��2@) �&~=�v@2�����1@q��D�]3@�������:              �?        �q���       �{�	�?�����A(*�
w
discriminator_loss*a	   @F`>   @F`>      �?!   @F`>) ��e��<2d�V�_>w&���qa>�������:              �?        
s
generator_loss*a	   `�2@   `�2@      �?!   `�2@)@�	Ɇv@2�����1@q��D�]3@�������:              �?        ���7�       �{�	K2a���AP*�
w
discriminator_loss*a	   �;�i>   �;�i>      �?!   �;�i>) �U��<2ڿ�ɓ�i>=�.^ol>�������:              �?        
s
generator_loss*a	   @�3@   @�3@      �?!   @�3@) �.԰�v@2�����1@q��D�]3@�������:              �?        �w��       �{�	�����Ax*�
w
discriminator_loss*a	   @�(e>   @�(e>      �?!   @�(e>) )>��<2cR�k�e>:�AC)8g>�������:              �?        
s
generator_loss*a	   @�3@   @�3@      �?!   @�3@) qpH�v@2�����1@q��D�]3@�������:              �?        o���       b�D�	�����A�*�
w
discriminator_loss*a	   ��O>   ��O>      �?!   ��O>) b	Ȳ*�<2������M>28���FP>�������:              �?        
s
generator_loss*a	   ��2@   ��2@      �?!   ��2@)@N�!C}v@2�����1@q��D�]3@�������:              �?        3�9��       b�D�	!�
���A�*�
w
discriminator_loss*a	    �{[>    �{[>      �?!    �{[>) ��ݚ�<24�j�6Z>��u}��\>�������:              �?        
s
generator_loss*a	   ��2@   ��2@      �?!   ��2@)@�*�[�v@2�����1@q��D�]3@�������:              �?        ���       b�D�	
�F���A�*�
w
discriminator_loss*a	    �W>    �W>      �?!    �W>) ���`��<2Fixі�W>4�j�6Z>�������:              �?        
s
generator_loss*a	    ?3@    ?3@      �?!    ?3@) x>/�v@2�����1@q��D�]3@�������:              �?        \�0��       b�D�	������A�*�
w
discriminator_loss*a	   @f�T>   @f�T>      �?!   @f�T>) q5�,��<2H��'ϱS>��x��U>�������:              �?        
s
generator_loss*a	   @=(3@   @=(3@      �?!   @=(3@) y*���v@2�����1@q��D�]3@�������:              �?        �����       b�D�	������A�*�
w
discriminator_loss*a	    �$Q>    �$Q>      �?!    �$Q>) @�Dy^�<228���FP>�
L�v�Q>�������:              �?        
s
generator_loss*a	   ��63@   ��63@      �?!   ��63@)@�hG�w@2�����1@q��D�]3@�������:              �?        �a|�       b�D�	15u���A�*�
w
discriminator_loss*a	   ���B>   ���B>      �?!   ���B>) d!�E��<2/�p`B>�`�}6D>�������:              �?        
s
generator_loss*a	   `cD3@   `cD3@      �?!   `cD3@)@6S�3w@2�����1@q��D�]3@�������:              �?        �`��       b�D�	^
6���A�*�
w
discriminator_loss*a	   ��?{>   ��?{>      �?!   ��?{>) �߰!4=2E'�/��x>f^��`{>�������:              �?        
s
generator_loss*a	   @WL3@   @WL3@      �?!   @WL3@) ɻv�Fw@2�����1@q��D�]3@�������:              �?        ���       b�D�	�,����A�*�
w
discriminator_loss*a	   @�)l>   @�)l>      �?!   @�)l>)��3�O��<2=�.^ol>w`f���n>�������:              �?        
s
generator_loss*a	   �iW3@   �iW3@      �?!   �iW3@)@� yaw@2�����1@q��D�]3@�������:              �?        �$?�       b�D�	.����A�*�
w
discriminator_loss*a	   �pBI>   �pBI>      �?!   �pBI>) ���Y�<2��8"uH>6��>?�J>�������:              �?        
s
generator_loss*a	    L[3@    L[3@      �?!    L[3@)  �q�jw@2�����1@q��D�]3@�������:              �?        ǽt5�       b�D�	�I ���A�*�
w
discriminator_loss*a	   �?Xi>   �?Xi>      �?!   �?Xi>) ��-��<2:�AC)8g>ڿ�ɓ�i>�������:              �?        
s
generator_loss*a	   �IJ3@   �IJ3@      �?!   �IJ3@) �1y�Aw@2�����1@q��D�]3@�������:              �?        ����       b�D�	�C#���A�*�
w
discriminator_loss*a	   `؂i>   `؂i>      �?!   `؂i>) ��V�<2:�AC)8g>ڿ�ɓ�i>�������:              �?        
s
generator_loss*a	   @V3@   @V3@      �?!   @V3@) �ND^w@2�����1@q��D�]3@�������:              �?        Ɛ���       b�D�	=�	&���A�*�
w
discriminator_loss*a	   �\fI>   �\fI>      �?!   �\fI>) b��7)�<2��8"uH>6��>?�J>�������:              �?        
s
generator_loss*a	    �e3@    �e3@      �?!    �e3@)  ���w@2q��D�]3@}w�˝M5@�������:              �?        �_�3�       b�D�	�r�(���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@|�?ZS=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	    �f3@    �f3@      �?!    �f3@)@�nφw@2q��D�]3@}w�˝M5@�������:              �?        2�%]�       b�D�	�\�+���A�*�
w
discriminator_loss*a	    J~o>    J~o>      �?!    J~o>)  kǡ��<2w`f���n>ہkVl�p>�������:              �?        
s
generator_loss*a	   ��k3@   ��k3@      �?!   ��k3@)@���ޒw@2q��D�]3@}w�˝M5@�������:              �?        ���H�       b�D�	iAf.���A�*�
w
discriminator_loss*a	   `BIx>   `BIx>      �?!   `BIx>) �_G�n=2�i����v>E'�/��x>�������:              �?        
s
generator_loss*a	   �^|3@   �^|3@      �?!   �^|3@) $�*'�w@2q��D�]3@}w�˝M5@�������:              �?        ����       b�D�	k%51���A�*�
w
discriminator_loss*a	   �{tV>   �{tV>      �?!   �{tV>) Dy����<2��x��U>Fixі�W>�������:              �?        
s
generator_loss*a	   `hl3@   `hl3@      �?!   `hl3@)@�gV�w@2q��D�]3@}w�˝M5@�������:              �?        ��¦�       b�D�	je4���A�*�
w
discriminator_loss*a	   ���W>   ���W>      �?!   ���W>) R>x���<2Fixі�W>4�j�6Z>�������:              �?        
s
generator_loss*a	   `.z3@   `.z3@      �?!   `.z3@)@j�'ӵw@2q��D�]3@}w�˝M5@�������:              �?        ?���       b�D�	���6���A�*�
w
discriminator_loss*a	   `�H�>   `�H�>      �?!   `�H�>) k"� =2T�L<�>��z!�?�>�������:              �?        
s
generator_loss*a	   `�k3@   `�k3@      �?!   `�k3@)@�V^�w@2q��D�]3@}w�˝M5@�������:              �?        g����       �N�	BƗ9���A*�
w
discriminator_loss*a	   @3�B>   @3�B>      �?!   @3�B>) )��&-�<2/�p`B>�`�}6D>�������:              �?        
s
generator_loss*a	   �'n3@   �'n3@      �?!   �'n3@)@`���w@2q��D�]3@}w�˝M5@�������:              �?        l}�       �{�	�r<���A(*�
w
discriminator_loss*a	   �dA|>   �dA|>      �?!   �dA|>) k���=2f^��`{>�����~>�������:              �?        
s
generator_loss*a	    ��3@    ��3@      �?!    ��3@) @�Nk�w@2q��D�]3@}w�˝M5@�������:              �?        �0�H�       �{�	6�J?���AP*�
w
discriminator_loss*a	   ���{>   ���{>      �?!   ���{>) )��=2f^��`{>�����~>�������:              �?        
s
generator_loss*a	   @v�3@   @v�3@      �?!   @v�3@) �1^��w@2q��D�]3@}w�˝M5@�������:              �?        ��Xs�       �{�	3#B���Ax*�
w
discriminator_loss*a	    rR>    rR>      �?!    rR>)@z_�D�<2�
L�v�Q>H��'ϱS>�������:              �?        
s
generator_loss*a	   ��v3@   ��v3@      �?!   ��v3@) �lڭw@2q��D�]3@}w�˝M5@�������:              �?        �}��       b�D�	i%�D���A�*�
w
discriminator_loss*a	   ��}>   ��}>      �?!   ��}>) "I�=2f^��`{>�����~>�������:              �?        
s
generator_loss*a	    �h3@    �h3@      �?!    �h3@) ����w@2q��D�]3@}w�˝M5@�������:              �?        p1���       b�D�	"�G���A�*�
w
discriminator_loss*a	   @�(}>   @�(}>      �?!   @�(}>)����{�
=2f^��`{>�����~>�������:              �?        
s
generator_loss*a	   ��u3@   ��u3@      �?!   ��u3@) Y���w@2q��D�]3@}w�˝M5@�������:              �?        IT�\�       b�D�	ǬJ���A�*�
w
discriminator_loss*a	    C�O>    C�O>      �?!    C�O>) HL�ȯ<2������M>28���FP>�������:              �?        
s
generator_loss*a	    ʂ3@    ʂ3@      �?!    ʂ3@)@h�*��w@2q��D�]3@}w�˝M5@�������:              �?        8_u3�       b�D�	�O�M���A�*�
w
discriminator_loss*a	    �B�>    �B�>      �?!    �B�>) �;��t=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   ���3@   ���3@      �?!   ���3@)@��N��w@2q��D�]3@}w�˝M5@�������:              �?        �b���       b�D�	aXmP���A�*�
w
discriminator_loss*a	   `n�U>   `n�U>      �?!   `n�U>)@j�]�<2��x��U>Fixі�W>�������:              �?        
s
generator_loss*a	   `��3@   `��3@      �?!   `��3@)@� |��w@2q��D�]3@}w�˝M5@�������:              �?        C�f�       b�D�	��NS���A�*�
w
discriminator_loss*a	   ���;>   ���;>      �?!   ���;>) �Ez �<2p
T~�;>����W_>>�������:              �?        
s
generator_loss*a	   ৘3@   ৘3@      �?!   ৘3@)@`�0? x@2q��D�]3@}w�˝M5@�������:              �?        ��#��       b�D�	�2V���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) BF&�bI=239W$:��>R%�����>�������:              �?        
s
generator_loss*a	   �o�3@   �o�3@      �?!   �o�3@) �d]�w@2q��D�]3@}w�˝M5@�������:              �?        �=�a�       b�D�	+6Y���A�*�
w
discriminator_loss*a	    o6s>    o6s>      �?!    o6s>) B� �<2BvŐ�r>�H5�8�t>�������:              �?        
s
generator_loss*a	   �`�3@   �`�3@      �?!   �`�3@) 	�%sx@2q��D�]3@}w�˝M5@�������:              �?        ;�N��       b�D�	79\���A�*�
w
discriminator_loss*a	    �Y>    �Y>      �?!    �Y>) 4J���<2Fixі�W>4�j�6Z>�������:              �?        
s
generator_loss*a	   ��3@   ��3@      �?!   ��3@)@�a�QMx@2q��D�]3@}w�˝M5@�������:              �?        A��D�       b�D�	T�^���A�*�
w
discriminator_loss*a	    $�q>    $�q>      �?!    $�q>)  Q���<2ہkVl�p>BvŐ�r>�������:              �?        
s
generator_loss*a	   ��3@   ��3@      �?!   ��3@) $���x@2q��D�]3@}w�˝M5@�������:              �?        �'�1�       b�D�	o\�a���A�*�
w
discriminator_loss*a	   �zY>   �zY>      �?!   �zY>) �wG��<2Fixі�W>4�j�6Z>�������:              �?        
s
generator_loss*a	   @f�3@   @f�3@      �?!   @f�3@) q�Y�x@2q��D�]3@}w�˝M5@�������:              �?        �`��       b�D�	A��d���A�*�
w
discriminator_loss*a	   @>.T>   @>.T>      �?!   @>.T>) 1�F!t�<2H��'ϱS>��x��U>�������:              �?        
s
generator_loss*a	   �$�3@   �$�3@      �?!   �$�3@) D3墫x@2q��D�]3@}w�˝M5@�������:              �?        ���       b�D�	�ȶg���A�*�
w
discriminator_loss*a	   @;�>   @;�>      �?!   @;�>)�����M=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	   ���3@   ���3@      �?!   ���3@) ���(�x@2q��D�]3@}w�˝M5@�������:              �?        ����       b�D�	��j���A�*�
w
discriminator_loss*a	   ���?>   ���?>      �?!   ���?>) ��d�(�<2����W_>>p��Dp�@>�������:              �?        
s
generator_loss*a	   @��3@   @��3@      �?!   @��3@) �7y�x@2q��D�]3@}w�˝M5@�������:              �?        M�<`�       b�D�	��m���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)��Y��&=2��ӤP��>�
�%W�>�������:              �?        
s
generator_loss*a	   ���3@   ���3@      �?!   ���3@) �
Q��x@2q��D�]3@}w�˝M5@�������:              �?        �}{%�       b�D�	��p���A�*�
w
discriminator_loss*a	   ���H>   ���H>      �?!   ���H>) $bU�<2��8"uH>6��>?�J>�������:              �?        
s
generator_loss*a	   �\�3@   �\�3@      �?!   �\�3@) �xΚx@2q��D�]3@}w�˝M5@�������:              �?        ���       b�D�	�vs���A�*�
w
discriminator_loss*a	    �_b>    �_b>      �?!    �_b>) �-��<2w&���qa>�����0c>�������:              �?        
s
generator_loss*a	   ���1@   ���1@      �?!   ���1@)@�Dt@2�����1@q��D�]3@�������:              �?        ��M��       b�D�	�@nv���A�*�
w
discriminator_loss*a	   �i�L>   �i�L>      �?!   �i�L>) L�L�A�<26��>?�J>������M>�������:              �?        
s
generator_loss*a	   @Ft2@   @Ft2@      �?!   @Ft2@) q��Hu@2�����1@q��D�]3@�������:              �?        �y���       �N�	��Iy���A*�
w
discriminator_loss*a	    �b>    �b>      �?!    �b>) ���{�<2w&���qa>�����0c>�������:              �?        
s
generator_loss*a	   ���2@   ���2@      �?!   ���2@) �d�u@2�����1@q��D�]3@�������:              �?        �Θh�       �{�	��H|���A(*�
w
discriminator_loss*a	   �#yi>   �#yi>      �?!   �#yi>) b�G�<2:�AC)8g>ڿ�ɓ�i>�������:              �?        
s
generator_loss*a	   ��3@   ��3@      �?!   ��3@) �����v@2�����1@q��D�]3@�������:              �?        8d��       �{�	�EF���AP*�
w
discriminator_loss*a	   �贁>   �贁>      �?!   �贁>) �l�=2[#=�؏�>K���7�>�������:              �?        
s
generator_loss*a	   �~E3@   �~E3@      �?!   �~E3@) $�F:6w@2�����1@q��D�]3@�������:              �?        �i�s�       �{�	�D����Ax*�
w
discriminator_loss*a	   ���u>   ���u>      �?!   ���u>) �����<2�H5�8�t>�i����v>�������:              �?        
s
generator_loss*a	   �o3@   �o3@      �?!   �o3@)@��~�w@2q��D�]3@}w�˝M5@�������:              �?        �hE�       b�D�	�J����A�*�
w
discriminator_loss*a	   ���/>   ���/>      �?!   ���/>) u@��o<27'_��+/>_"s�$1>�������:              �?        
s
generator_loss*a	   @��3@   @��3@      �?!   @��3@) Igbx@2q��D�]3@}w�˝M5@�������:              �?        2����       b�D�	j�L����A�*�
w
discriminator_loss*a	   ���*>   ���*>      �?!   ���*>)���{�0f<2���<�)>�'v�V,>�������:              �?        
s
generator_loss*a	    ��3@    ��3@      �?!    ��3@) @�`�{x@2q��D�]3@}w�˝M5@�������:              �?        h��       b�D�	�S����A�*�
w
discriminator_loss*a	   ��$)>   ��$)>      �?!   ��$)>) ��F|�c<24��evk'>���<�)>�������:              �?        
s
generator_loss*a	   `��3@   `��3@      �?!   `��3@)@��[��x@2q��D�]3@}w�˝M5@�������:              �?        ���       b�D�	xY����A�*�
w
discriminator_loss*a	    �E^>    �E^>      �?!    �E^>) za%��<2��u}��\>d�V�_>�������:              �?        
s
generator_loss*a	    �4@    �4@      �?!    �4@) @63yy@2q��D�]3@}w�˝M5@�������:              �?        ?�C��       b�D�	�`����A�*�
w
discriminator_loss*a	    6!�>    6!�>      �?!    6!�>) ��/A�(=2��ӤP��>�
�%W�>�������:              �?        
s
generator_loss*a	   �4@   �4@      �?!   �4@)@��=!(y@2q��D�]3@}w�˝M5@�������:              �?        lb�q�       b�D�	^�g����A�*�
w
discriminator_loss*a	   ��*>   ��*>      �?!   ��*>) �;U�Ce<2���<�)>�'v�V,>�������:              �?        
s
generator_loss*a	    �4@    �4@      �?!    �4@)@�g�?*y@2q��D�]3@}w�˝M5@�������:              �?        �C�f�       b�D�	��m����A�*�
w
discriminator_loss*a	   ���0>   ���0>      �?!   ���0>) ���#�q<27'_��+/>_"s�$1>�������:              �?        
s
generator_loss*a	   ��!4@   ��!4@      �?!   ��!4@)@N1�Ty@2q��D�]3@}w�˝M5@�������:              �?        ��{�       b�D�	�.u����A�*�
w
discriminator_loss*a	   �֟S>   �֟S>      �?!   �֟S>)@����<2�
L�v�Q>H��'ϱS>�������:              �?        
s
generator_loss*a	   `�3@   `�3@      �?!   `�3@)@��t��x@2q��D�]3@}w�˝M5@�������:              �?        ].�       b�D�	�~����A�*�
w
discriminator_loss*a	   �F�{>   �F�{>      �?!   �F�{>) R�w�|=2f^��`{>�����~>�������:              �?        
s
generator_loss*a	   �14@   �14@      �?!   �14@)@�	[�y@2q��D�]3@}w�˝M5@�������:              �?        C��N�       b�D�	a:�����A�*�
w
discriminator_loss*a	   @�~>   @�~>      �?!   @�~>)���*�$=2f^��`{>�����~>�������:              �?        
s
generator_loss*a	   ��4@   ��4@      �?!   ��4@) �e$y@2q��D�]3@}w�˝M5@�������:              �?        ���       b�D�	�袣���A�*�
w
discriminator_loss*a	   ���a>   ���a>      �?!   ���a>) 1\k���<2w&���qa>�����0c>�������:              �?        
s
generator_loss*a	   @��3@   @��3@      �?!   @��3@) �3 ��x@2q��D�]3@}w�˝M5@�������:              �?        r���       b�D�	�������A�*�
w
discriminator_loss*a	   ���&>   ���&>      �?!   ���&>) �3�x:`<2��o�kJ%>4��evk'>�������:              �?        
s
generator_loss*a	   ��4@   ��4@      �?!   ��4@) yjfv'y@2q��D�]3@}w�˝M5@�������:              �?        ��       b�D�	�Iܩ���A�*�
w
discriminator_loss*a	    	&1>    	&1>      �?!    	&1>) Ŋ-ar<2_"s�$1>6NK��2>�������:              �?        
s
generator_loss*a	   @�4@   @�4@      �?!   @�4@) Ѩ	�Ly@2q��D�]3@}w�˝M5@�������:              �?        y����       b�D�	�b�����A�*�
w
discriminator_loss*a	    3�d>    3�d>      �?!    3�d>) ���=��<2�����0c>cR�k�e>�������:              �?        
s
generator_loss*a	    l24@    l24@      �?!    l24@)  ��~y@2q��D�]3@}w�˝M5@�������:              �?        KC��       b�D�	5����A�*�
w
discriminator_loss*a	   @��c>   @��c>      �?!   @��c>) �e�	��<2�����0c>cR�k�e>�������:              �?        
s
generator_loss*a	   @j64@   @j64@      �?!   @j64@) ��y@2q��D�]3@}w�˝M5@�������:              �?        ����       b�D�	�,����A�*�
w
discriminator_loss*a	   �Q�=>   �Q�=>      �?!   �Q�=>) ��V��<2p
T~�;>����W_>>�������:              �?        
s
generator_loss*a	   �~84@   �~84@      �?!   �~84@) ,[�y@2q��D�]3@}w�˝M5@�������:              �?        �S�v�       b�D�	�B����A�*�
w
discriminator_loss*a	    ��4>    ��4>      �?!    ��4>)@d;T��z<2�so쩾4>�z��6>�������:              �?        
s
generator_loss*a	   ��34@   ��34@      �?!   ��34@) я|��y@2q��D�]3@}w�˝M5@�������:              �?        ´��       b�D�	]����A�*�
w
discriminator_loss*a	   ���(>   ���(>      �?!   ���(>) �kU5c<24��evk'>���<�)>�������:              �?        
s
generator_loss*a	    \24@    \24@      �?!    \24@)  ��~y@2q��D�]3@}w�˝M5@�������:              �?        �*Ǣ�       �N�	�a����A*�
w
discriminator_loss*a	   @r3@>   @r3@>      �?!   @r3@>) ч�g�<2����W_>>p��Dp�@>�������:              �?        
s
generator_loss*a	    X94@    X94@      �?!    X94@)@`��)�y@2q��D�]3@}w�˝M5@�������:              �?        �����       �{�	������A(*�
w
discriminator_loss*a	   ��x>   ��x>      �?!   ��x>) B���*=2�i����v>E'�/��x>�������:              �?        
s
generator_loss*a	    �D4@    �D4@      �?!    �D4@) @���y@2q��D�]3@}w�˝M5@�������:              �?        vc���       �{�	�X�����AP*�
w
discriminator_loss*a	    �"[>    �"[>      �?!    �"[>) �R��<24�j�6Z>��u}��\>�������:              �?        
s
generator_loss*a	    �64@    �64@      �?!    �64@)@|���y@2q��D�]3@}w�˝M5@�������:              �?        �0x�       �{�	#������Ax*�
w
discriminator_loss*a	   ���U>   ���U>      �?!   ���U>) �Z�9�<2H��'ϱS>��x��U>�������:              �?        
s
generator_loss*a	    �:4@    �:4@      �?!    �:4@)  �{�y@2q��D�]3@}w�˝M5@�������:              �?        ����       b�D�	�������A�*�
w
discriminator_loss*a	   �)3F>   �)3F>      �?!   �)3F>) �jV͞<2�`�}6D>��Ő�;F>�������:              �?        
s
generator_loss*a	   ��G4@   ��G4@      �?!   ��G4@)@�P}�y@2q��D�]3@}w�˝M5@�������:              �?        1Z��       b�D�	Ԩ	����A�*�
w
discriminator_loss*a	   ��J>   ��J>      �?!   ��J>)�#�:K�<2��8"uH>6��>?�J>�������:              �?        
s
generator_loss*a	    $[4@    $[4@      �?!    $[4@)  �)��y@2q��D�]3@}w�˝M5@�������:              �?        ��J�       b�D�	�I3����A�*�
w
discriminator_loss*a	    ��.>    ��.>      �?!    ��.>)  ǋ�m<2�'v�V,>7'_��+/>�������:              �?        
s
generator_loss*a	   ��s4@   ��s4@      �?!   ��s4@) A�I�$z@2q��D�]3@}w�˝M5@�������:              �?        δ#��       b�D�	�jb����A�*�
w
discriminator_loss*a	   ���J>   ���J>      �?!   ���J>) R6�O��<26��>?�J>������M>�������:              �?        
s
generator_loss*a	   ��4@   ��4@      �?!   ��4@) �I4Cz@2q��D�]3@}w�˝M5@�������:              �?        w`oY�       b�D�	������A�*�
w
discriminator_loss*a	   `qy�>   `qy�>      �?!   `qy�>)@^ﱪS=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   �΃4@   �΃4@      �?!   �΃4@)@��Mz@2q��D�]3@}w�˝M5@�������:              �?        �i��       b�D�	߾����A�*�
w
discriminator_loss*a	   �x9>   �x9>      �?!   �x9>) ��fE�<2u 5�9>p
T~�;>�������:              �?        
s
generator_loss*a	   ��4@   ��4@      �?!   ��4@) $*�`z@2q��D�]3@}w�˝M5@�������:              �?        ��Gz�       b�D�	�	�����A�*�
w
discriminator_loss*a	   ๺�>   ๺�>      �?!   ๺�>) �)%`=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   ��4@   ��4@      �?!   ��4@)@�@�ez@2q��D�]3@}w�˝M5@�������:              �?        �s���       b�D�	K�����A�*�
w
discriminator_loss*a	   ��؝>   ��؝>      �?!   ��؝>) R"�h�K=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	    ��4@    ��4@      �?!    ��4@)@�H�sz@2q��D�]3@}w�˝M5@�������:              �?        �;V��       b�D�	Q�G����A�*�
w
discriminator_loss*a	   @v<�>   @v<�>      �?!   @v<�>)��$e��%=2��z!�?�>��ӤP��>�������:              �?        
s
generator_loss*a	    �4@    �4@      �?!    �4@)@Ե�z@2q��D�]3@}w�˝M5@�������:              �?        �����       b�D�	u����A�*�
w
discriminator_loss*a	   @��T>   @��T>      �?!   @��T>) Y���=�<2H��'ϱS>��x��U>�������:              �?        
s
generator_loss*a	   ཟ4@   ཟ4@      �?!   ཟ4@)@H����z@2q��D�]3@}w�˝M5@�������:              �?        W�[��       b�D�	�r�����A�*�
w
discriminator_loss*a	   ���=>   ���=>      �?!   ���=>) B�xI��<2p
T~�;>����W_>>�������:              �?        
s
generator_loss*a	    Ϧ4@    Ϧ4@      �?!    Ϧ4@) ��Чz@2q��D�]3@}w�˝M5@�������:              �?        �����       b�D�	�X�����A�*�
w
discriminator_loss*a	   �"a4>   �"a4>      �?!   �"a4>) y��$�y<26NK��2>�so쩾4>�������:              �?        
s
generator_loss*a	    ?�4@    ?�4@      �?!    ?�4@) X�"�z@2q��D�]3@}w�˝M5@�������:              �?        �Փ��       b�D�	Um����A�*�
w
discriminator_loss*a	   � +0>   � +0>      �?!   � +0>) �>�Vp<27'_��+/>_"s�$1>�������:              �?        
s
generator_loss*a	   @?�4@   @?�4@      �?!   @?�4@) 	:Nh�z@2q��D�]3@}w�˝M5@�������:              �?        �_��       b�D�	]OK����A�*�
w
discriminator_loss*a	    ��*>    ��*>      �?!    ��*>) ����e<2���<�)>�'v�V,>�������:              �?        
s
generator_loss*a	   �+�4@   �+�4@      �?!   �+�4@) ���?�z@2q��D�]3@}w�˝M5@�������:              �?        ��m�       b�D�	Z{�����A�*�
w
discriminator_loss*a	   ���L>   ���L>      �?!   ���L>)��]M©<26��>?�J>������M>�������:              �?        
s
generator_loss*a	   ��4@   ��4@      �?!   ��4@) �����z@2q��D�]3@}w�˝M5@�������:              �?        =J��       b�D�	�)�����A�*�
w
discriminator_loss*a	   ���F>   ���F>      �?!   ���F>) �$�t��<2��Ő�;F>��8"uH>�������:              �?        
s
generator_loss*a	   �)�4@   �)�4@      �?!   �)�4@) ����z@2q��D�]3@}w�˝M5@�������:              �?        ��Ϻ�       b�D�	b�	����A�*�
w
discriminator_loss*a	    m�1>    m�1>      �?!    m�1>)@DL�Bt<2_"s�$1>6NK��2>�������:              �?        
s
generator_loss*a	   ��4@   ��4@      �?!   ��4@)@j�ڊ {@2q��D�]3@}w�˝M5@�������:              �?        ��       b�D�	״M����A�*�
w
discriminator_loss*a	   �c7>   �c7>      �?!   �c7>)�0@&��<2�z��6>u 5�9>�������:              �?        
s
generator_loss*a	   ���4@   ���4@      �?!   ���4@)@����>{@2q��D�]3@}w�˝M5@�������:              �?        �n�       �N�		�t���A*�
w
discriminator_loss*a	    �>    �>      �?!    �>)  ���G<2��f��p>�i
�k>�������:              �?        
s
generator_loss*a	   `_�4@   `_�4@      �?!   `_�4@)@��0_{@2q��D�]3@}w�˝M5@�������:              �?        w�_��       �{�	�f����A(*�
w
discriminator_loss*a	   �, &>   �, &>      �?!   �, &>)@v�j��^<2��o�kJ%>4��evk'>�������:              �?        
s
generator_loss*a	    ��4@    ��4@      �?!    ��4@)@�n�Rb{@2q��D�]3@}w�˝M5@�������:              �?        �?��       �{�	�	���AP*�
w
discriminator_loss*a	   �PC=>   �PC=>      �?!   �PC=>) fv1�<2p
T~�;>����W_>>�������:              �?        
s
generator_loss*a	    �4@    �4@      �?!    �4@)@���i{@2q��D�]3@}w�˝M5@�������:              �?        /���       �{�	�uQ���Ax*�
w
discriminator_loss*a	    >{#>    >{#>      �?!    >{#>) @0�h�W<24�e|�Z#>��o�kJ%>�������:              �?        
s
generator_loss*a	   �]�4@   �]�4@      �?!   �]�4@)@Ȗ%�G{@2q��D�]3@}w�˝M5@�������:              �?        6.��       b�D�	�ܚ���A�*�
w
discriminator_loss*a	   `�e>   `�e>      �?!   `�e>)@��x�<2cR�k�e>:�AC)8g>�������:              �?        
s
generator_loss*a	   @��4@   @��4@      �?!   @��4@) �7�A{@2q��D�]3@}w�˝M5@�������:              �?        �X�C�       b�D�	ޮ����A�*�
w
discriminator_loss*a	   �z�#>   �z�#>      �?!   �z�#>)@�+�p�X<24�e|�Z#>��o�kJ%>�������:              �?        
s
generator_loss*a	   ���4@   ���4@      �?!   ���4@)@�	�;{@2q��D�]3@}w�˝M5@�������:              �?        �A9?�       b�D�	�6���A�*�
w
discriminator_loss*a	   ��~J>   ��~J>      �?!   ��~J>)��H$�<2��8"uH>6��>?�J>�������:              �?        
s
generator_loss*a	    ��4@    ��4@      �?!    ��4@)@H�/�Z{@2q��D�]3@}w�˝M5@�������:              �?        f�;��       b�D�	~w����A�*�
w
discriminator_loss*a	   �or>   �or>      �?!   �or>)  �+<2���">Z�TA[�>�������:              �?        
s
generator_loss*a	   ���4@   ���4@      �?!   ���4@) ��Qu{@2q��D�]3@}w�˝M5@�������:              �?        �_�       b�D�	u0����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) $�V��G<2��f��p>�i
�k>�������:              �?        
s
generator_loss*a	   @5@   @5@      �?!   @5@) 	u�H�{@2q��D�]3@}w�˝M5@�������:              �?        (�_��       b�D�	��5 ���A�*�
w
discriminator_loss*a	   ��M>   ��M>      �?!   ��M>)��b����<2������M>28���FP>�������:              �?        
s
generator_loss*a	   ��5@   ��5@      �?!   ��5@) �š��{@2q��D�]3@}w�˝M5@�������:              �?        w�}��       b�D�	��#���A�*�
w
discriminator_loss*a	   `�KA>   `�KA>      �?!   `�KA>)@�o���<2p��Dp�@>/�p`B>�������:              �?        
s
generator_loss*a	   @o5@   @o5@      �?!   @o5@) �}3�{@2q��D�]3@}w�˝M5@�������:              �?        0�|�       b�D�	,u�&���A�*�
w
discriminator_loss*a	    �~3>    �~3>      �?!    �~3>)@tq]��w<26NK��2>�so쩾4>�������:              �?        
s
generator_loss*a	    2.5@    2.5@      �?!    2.5@) @��	|@2q��D�]3@}w�˝M5@�������:              �?        سE��       b�D�	'�9*���A�*�
w
discriminator_loss*a	   `�o>   `�o>      �?!   `�o>) ��.�<2w`f���n>ہkVl�p>�������:              �?        
s
generator_loss*a	   `�>5@   `�>5@      �?!   `�>5@)@�z��5|@2q��D�]3@}w�˝M5@�������:              �?        �Բ��       b�D�	�e�-���A�*�
w
discriminator_loss*a	   @b�D>   @b�D>      �?!   @b�D>) Q�a��<2�`�}6D>��Ő�;F>�������:              �?        
s
generator_loss*a	   �*M5@   �*M5@      �?!   �*M5@)@��\|@2q��D�]3@}w�˝M5@�������:              �?        ��9�       b�D�	�t�0���A�*�
w
discriminator_loss*a	   ���<>   ���<>      �?!   ���<>) �zI#�<2p
T~�;>����W_>>�������:              �?        
s
generator_loss*a	    kK5@    kK5@      �?!    kK5@)@<�\W|@2q��D�]3@}w�˝M5@�������:              �?        2����       b�D�	�yG4���A�*�
w
discriminator_loss*a	   �cH1>   �cH1>      �?!   �cH1>) ����r<2_"s�$1>6NK��2>�������:              �?        
s
generator_loss*a	   �<H5@   �<H5@      �?!   �<H5@)@��F�N|@2q��D�]3@}w�˝M5@�������:              �?        �70`�       b�D�	��7���A�*�
w
discriminator_loss*a	   �I�5>   �I�5>      �?!   �I�5>)@���8~<2�so쩾4>�z��6>�������:              �?        
s
generator_loss*a	    z5@    z5@      �?!    z5@)@(X��{@2q��D�]3@}w�˝M5@�������:              �?        �Z�       b�D�	];���A�*�
w
discriminator_loss*a	   �g>;>   �g>;>      �?!   �g>;>)�`X��1�<2u 5�9>p
T~�;>�������:              �?        
s
generator_loss*a	   �p#5@   �p#5@      �?!   �p#5@) G�U�{@2q��D�]3@}w�˝M5@�������:              �?        �T�5�       b�D�	^Bs>���A�*�
w
discriminator_loss*a	   ���P>   ���P>      �?!   ���P>) dto8��<228���FP>�
L�v�Q>�������:              �?        
s
generator_loss*a	   �J'5@   �J'5@      �?!   �J'5@) ���{@2q��D�]3@}w�˝M5@�������:              �?        ���       b�D�	��A���A�*�
w
discriminator_loss*a	   �[�>   �[�>      �?!   �[�>)@�@�4<2�#���j>�J>�������:              �?        
s
generator_loss*a	   ��35@   ��35@      �?!   ��35@) ?)|@2q��D�]3@}w�˝M5@�������:              �?        �ɩ�       b�D�	��5E���A�*�
w
discriminator_loss*a	   ���1>   ���1>      �?!   ���1>)@N�s<2_"s�$1>6NK��2>�������:              �?        
s
generator_loss*a	   @eF5@   @eF5@      �?!   @eF5@) �p��I|@2q��D�]3@}w�˝M5@�������:              �?        ����       b�D�	³�H���A�*�
w
discriminator_loss*a	   �� >   �� >      �?!   �� >) $����6<2�J>2!K�R�>�������:              �?        
s
generator_loss*a	    �W5@    �W5@      �?!    �W5@)@쫓!x|@2}w�˝M5@�i*`�n7@�������:              �?        ��%�       �N�	w��K���A*�
w
discriminator_loss*a	   �x?>   �x?>      �?!   �x?>) ����<2����W_>>p��Dp�@>�������:              �?        
s
generator_loss*a	   `�U5@   `�U5@      �?!   `�U5@)@�-��r|@2}w�˝M5@�i*`�n7@�������:              �?        $h]��       �{�	o(OO���A(*�
w
discriminator_loss*a	   ��AD>   ��AD>      �?!   ��AD>) Qp�
��<2�`�}6D>��Ő�;F>�������:              �?        
s
generator_loss*a	   �<U5@   �<U5@      �?!   �<U5@)@�	9�q|@2}w�˝M5@�i*`�n7@�������:              �?        `�S��       �{�	)h�R���AP*�
w
discriminator_loss*a	    �'�>    �'�>      �?!    �'�>)@���6=2�4[_>��>
�}���>�������:              �?        
s
generator_loss*a	   �?f5@   �?f5@      �?!   �?f5@) ���|@2}w�˝M5@�i*`�n7@�������:              �?        I#���       �{�	��(V���Ax*�
w
discriminator_loss*a	    �=*>    �=*>      �?!    �=*>) �>�݄e<2���<�)>�'v�V,>�������:              �?        
s
generator_loss*a	   ��E5@   ��E5@      �?!   ��E5@) ��aH|@2q��D�]3@}w�˝M5@�������:              �?        �L��       b�D�	���Y���A�*�
w
discriminator_loss*a	   @		8>   @		8>      �?!   @		8>)��6m��<2�z��6>u 5�9>�������:              �?        
s
generator_loss*a	   �Z5@   �Z5@      �?!   �Z5@)@x��|@2}w�˝M5@�i*`�n7@�������:              �?        <3���       b�D�	�>]���A�*�
w
discriminator_loss*a	   �v�>   �v�>      �?!   �v�>) Y���1<2�#���j>�J>�������:              �?        
s
generator_loss*a	   �!y5@   �!y5@      �?!   �!y5@) 1����|@2}w�˝M5@�i*`�n7@�������:              �?        z���       b�D�	��`���A�*�
w
discriminator_loss*a	    ya>    ya>      �?!    ya>) ��C�0<2Z�TA[�>�#���j>�������:              �?        
s
generator_loss*a	   �y�5@   �y�5@      �?!   �y�5@)@XTKX�|@2}w�˝M5@�i*`�n7@�������:              �?        �~��       b�D�	��c���A�*�
w
discriminator_loss*a	   �i!>   �i!>      �?!   �i!>) ��aR<2%���>��-�z�!>�������:              �?        
s
generator_loss*a	   �g�5@   �g�5@      �?!   �g�5@) �}S�}@2}w�˝M5@�i*`�n7@�������:              �?        ���       b�D�	�tg���A�*�
w
discriminator_loss*a	   �P>   �P>      �?!   �P>) 	,F�E<2Łt�=	>��f��p>�������:              �?        
s
generator_loss*a	    2�5@    2�5@      �?!    2�5@)@Q��2}@2}w�˝M5@�i*`�n7@�������:              �?        �s���       b�D�	�=�j���A�*�
w
discriminator_loss*a	    >    >      �?!    >) ���@<2��R���>Łt�=	>�������:              �?        
s
generator_loss*a	   `��5@   `��5@      �?!   `��5@)@VY��?}@2}w�˝M5@�i*`�n7@�������:              �?        �a��       b�D�	�Jhn���A�*�
w
discriminator_loss*a	    f�>    f�>      �?!    f�>) ��`�K<2�i
�k>%���>�������:              �?        
s
generator_loss*a	   @s�5@   @s�5@      �?!   @s�5@) )���y}@2}w�˝M5@�i*`�n7@�������:              �?        z���       b�D�		��q���A�*�
w
discriminator_loss*a	    �N>    �N>      �?!    �N>) h�Y�.�<2������M>28���FP>�������:              �?        
s
generator_loss*a	   @q�5@   @q�5@      �?!   @q�5@) ���	w}@2}w�˝M5@�i*`�n7@�������:              �?        �d���       b�D�	d�bu���A�*�
w
discriminator_loss*a	   `��9>   `��9>      �?!   `��9>) �ߡ�Ʉ<2u 5�9>p
T~�;>�������:              �?        
s
generator_loss*a	    ��5@    ��5@      �?!    ��5@) ��׏}@2}w�˝M5@�i*`�n7@�������:              �?        ���|�       b�D�	-?�x���A�*�
w
discriminator_loss*a	    �s>    �s>      �?!    �s>)@�}F�<2BvŐ�r>�H5�8�t>�������:              �?        
s
generator_loss*a	   `��5@   `��5@      �?!   `��5@)@ʖ'<W}@2}w�˝M5@�i*`�n7@�������:              �?        m�.��       b�D�	ۓd|���A�*�
w
discriminator_loss*a	   �P�>   �P�>      �?!   �P�>)@FfcoN6=2�4[_>��>
�}���>�������:              �?        
s
generator_loss*a	   ��n5@   ��n5@      �?!   ��n5@)@����|@2}w�˝M5@�i*`�n7@�������:              �?        �)�/�       b�D�	�	����A�*�
w
discriminator_loss*a	   �:&1>   �:&1>      �?!   �:&1>) 䵦�ar<2_"s�$1>6NK��2>�������:              �?        
s
generator_loss*a	   ��t5@   ��t5@      �?!   ��t5@)@"l���|@2}w�˝M5@�i*`�n7@�������:              �?        ML%p�       b�D�	f�h����A�*�
w
discriminator_loss*a	   �u>   �u>      �?!   �u>) �
+'0<2Z�TA[�>�#���j>�������:              �?        
s
generator_loss*a	    n�5@    n�5@      �?!    n�5@) @t5J�|@2}w�˝M5@�i*`�n7@�������:              �?        �����       b�D�	�y�����A�*�
w
discriminator_loss*a	   દ>   દ>      �?!   દ>)@��<�x3<2�#���j>�J>�������:              �?        
s
generator_loss*a	   �0�5@   �0�5@      �?!   �0�5@)@�G���|@2}w�˝M5@�i*`�n7@�������:              �?        �L�@�       b�D�	m~����A�*�
w
discriminator_loss*a	    �d>    �d>      �?!    �d>) �2?> K<2�i
�k>%���>�������:              �?        
s
generator_loss*a	   ���5@   ���5@      �?!   ���5@) dC��}@2}w�˝M5@�i*`�n7@�������:              �?        ����       b�D�	v�����A�*�
w
discriminator_loss*a	   �G�3>   �G�3>      �?!   �G�3>)@� E@:x<26NK��2>�so쩾4>�������:              �?        
s
generator_loss*a	    ץ5@    ץ5@      �?!    ץ5@)@l@�J}@2}w�˝M5@�i*`�n7@�������:              �?        1� �       b�D�	������A�*�
w
discriminator_loss*a	   ��j >   ��j >      �?!   ��j >)@�a�x�P<2%���>��-�z�!>�������:              �?        
s
generator_loss*a	   ���5@   ���5@      �?!   ���5@) �{�\}@2}w�˝M5@�i*`�n7@�������:              �?        �	���       b�D�	������A�*�
w
discriminator_loss*a	   @�H>   @�H>      �?!   @�H>) !����9<22!K�R�>��R���>�������:              �?        
s
generator_loss*a	   �+�5@   �+�5@      �?!   �+�5@)@��8}@2}w�˝M5@�i*`�n7@�������:              �?        mp}L�       �N�	�f�����A*�
w
discriminator_loss*a	   `B5>   `B5>      �?!   `B5>)@Z�Tށ<2nx6�X� >�`��>�������:              �?        
s
generator_loss*a	   � �5@   � �5@      �?!   � �5@) �[X}@2}w�˝M5@�i*`�n7@�������:              �?        �]���       �{�	�!����A(*�
w
discriminator_loss*a	   �[B(>   �[B(>      �?!   �[B(>) �u�db<24��evk'>���<�)>�������:              �?        
s
generator_loss*a	   `n�5@   `n�5@      �?!   `n�5@)@j�s��}@2}w�˝M5@�i*`�n7@�������:              �?        Ǫ���       �{�	������AP*�
w
discriminator_loss*a	   ��&>   ��&>      �?!   ��&>)�l��QC`<2��o�kJ%>4��evk'>�������:              �?        
s
generator_loss*a	   �f�5@   �f�5@      �?!   �f�5@) �Ã�}@2}w�˝M5@�i*`�n7@�������:              �?        ��9��       �{�	�q6����Ax*�
w
discriminator_loss*a	   ���&>   ���&>      �?!   ���&>) �\�$�`<2��o�kJ%>4��evk'>�������:              �?        
s
generator_loss*a	   @�5@   @�5@      �?!   @�5@) �\,�}@2}w�˝M5@�i*`�n7@�������:              �?        m�R�       b�D�	�ͦ���A�*�
w
discriminator_loss*a	   ��&>   ��&>      �?!   ��&>)@�Q`B�_<2��o�kJ%>4��evk'>�������:              �?        
s
generator_loss*a	   �=�5@   �=�5@      �?!   �=�5@) d�ı~@2}w�˝M5@�i*`�n7@�������:              �?        �-S��       b�D�	<�d����A�*�
w
discriminator_loss*a	   ��>>   ��>>      �?!   ��>>) "�׹J<2�i
�k>%���>�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@) ��4fJ~@2}w�˝M5@�i*`�n7@�������:              �?        �vX�       b�D�	�������A�*�
w
discriminator_loss*a	    |F>    |F>      �?!    |F>) �`@��(<2���">Z�TA[�>�������:              �?        
s
generator_loss*a	   @6@   @6@      �?!   @6@) �P�0V~@2}w�˝M5@�i*`�n7@�������:              �?        �Ꮲ�       b�D�	������A�*�
w
discriminator_loss*a	   �1R>   �1R>      �?!   �1R>)@:�<��<2�
L�v�Q>H��'ϱS>�������:              �?        
s
generator_loss*a	   �B6@   �B6@      �?!   �B6@)@�+��H~@2}w�˝M5@�i*`�n7@�������:              �?        ��W��       b�D�	"����A�*�
w
discriminator_loss*a	    ��4>    ��4>      �?!    ��4>) @��J{<2�so쩾4>�z��6>�������:              �?        
s
generator_loss*a	   ��	6@   ��	6@      �?!   ��	6@)@ ��[~@2}w�˝M5@�i*`�n7@�������:              �?        �M��       b�D�	�n�����A�*�
w
discriminator_loss*a	   �� >   �� >      �?!   �� >)@���#P<2%���>��-�z�!>�������:              �?        
s
generator_loss*a	   `��5@   `��5@      �?!   `��5@)@�&�z&~@2}w�˝M5@�i*`�n7@�������:              �?        �+g��       b�D�	�[����A�*�
w
discriminator_loss*a	    c�T>    c�T>      �?!    c�T>)@�Z���<2H��'ϱS>��x��U>�������:              �?        
s
generator_loss*a	   ��5@   ��5@      �?!   ��5@)@���?~@2}w�˝M5@�i*`�n7@�������:              �?        ��)��       b�D�	������A�*�
w
discriminator_loss*a	    �~(>    �~(>      �?!    �~(>) d�O�b<24��evk'>���<�)>�������:              �?        
s
generator_loss*a	   @�6@   @�6@      �?!   @�6@) I\�O~@2}w�˝M5@�i*`�n7@�������:              �?        uF��       b�D�	'˗����A�*�
w
discriminator_loss*a	    c�>    c�>      �?!    c�>) H�Zl�G<2��f��p>�i
�k>�������:              �?        
s
generator_loss*a	    ��5@    ��5@      �?!    ��5@)  ���}@2}w�˝M5@�i*`�n7@�������:              �?        ���       b�D�	]:����A�*�
w
discriminator_loss*a	   @��	>   @��	>      �?!   @��	>)������$<2RT��+�>���">�������:              �?        
s
generator_loss*a	    4�5@    4�5@      �?!    4�5@)@�A�8�}@2}w�˝M5@�i*`�n7@�������:              �?        k����       b�D�	�2�����A�*�
w
discriminator_loss*a	   ����=   ����=      �?!   ����=) Zla�<2�f׽r��=nx6�X� >�������:              �?        
s
generator_loss*a	   `�6@   `�6@      �?!   `�6@)@f(hU~@2}w�˝M5@�i*`�n7@�������:              �?        Q���       b�D�	"~����A�*�
w
discriminator_loss*a	   @���=   @���=      �?!   @���=)�@��~<2f;H�\Q�=�tO���=�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@) �pbr�~@2}w�˝M5@�i*`�n7@�������:              �?        �8�M�       b�D�	2�(����A�*�
w
discriminator_loss*a	   `F '>   `F '>      �?!   `F '>) Ś*e�`<2��o�kJ%>4��evk'>�������:              �?        
s
generator_loss*a	    �*6@    �*6@      �?!    �*6@)@�\Rf�~@2}w�˝M5@�i*`�n7@�������:              �?        h5�@�       b�D�	ӵ�����A�*�
w
discriminator_loss*a	   �vX#>   �vX#>      �?!   �vX#>)@4sdW<2��-�z�!>4�e|�Z#>�������:              �?        
s
generator_loss*a	   `<66@   `<66@      �?!   `<66@)@�k���~@2}w�˝M5@�i*`�n7@�������:              �?        M20��       b�D�	a�v����A�*�
w
discriminator_loss*a	   �;:>   �;:>      �?!   �;:>)@�&"<2�`��>�mm7&c>�������:              �?        
s
generator_loss*a	   �r<6@   �r<6@      �?!   �r<6@)@��H �~@2}w�˝M5@�i*`�n7@�������:              �?        P��_�       b�D�	�J ����A�*�
w
discriminator_loss*a	   @&� >   @&� >      �?!   @&� >) q"U�Q<2%���>��-�z�!>�������:              �?        
s
generator_loss*a	   ��/6@   ��/6@      �?!   ��/6@) $X��~@2}w�˝M5@�i*`�n7@�������:              �?        ���~�       b�D�	v_�����A�*�
w
discriminator_loss*a	   ��u>   ��u>      �?!   ��u>) ����<<22!K�R�>��R���>�������:              �?        
s
generator_loss*a	   `v+6@   `v+6@      �?!   `v+6@)@ʯ���~@2}w�˝M5@�i*`�n7@�������:              �?        �� �       b�D�	�a�����A�*�
w
discriminator_loss*a	    ��^>    ��^>      �?!    ��^>) �څ\�<2��u}��\>d�V�_>�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@)@��`T�~@2}w�˝M5@�i*`�n7@�������:              �?        �����       �N�	1����A*�
w
discriminator_loss*a	   @4>   @4>      �?!   @4>)�P�\�<.<2Z�TA[�>�#���j>�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@)@J�V�`~@2}w�˝M5@�i*`�n7@�������:              �?        5h�H�       �{�	������A(*�
w
discriminator_loss*a	   �	U�>   �	U�>      �?!   �	U�>)����A=2X$�z�>.��fc��>�������:              �?        
s
generator_loss*a	   @�6@   @�6@      �?!   @�6@) IV�s�~@2}w�˝M5@�i*`�n7@�������:              �?        F�@J�       �{�	躍����AP*�
w
discriminator_loss*a	   ���<>   ���<>      �?!   ���<>) �զ$7�<2p
T~�;>����W_>>�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@)@����J~@2}w�˝M5@�i*`�n7@�������:              �?        !3���       �{�	W�D����Ax*�
w
discriminator_loss*a	    Պk>    Պk>      �?!    Պk>) ȩ¡��<2ڿ�ɓ�i>=�.^ol>�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@) ����~@2}w�˝M5@�i*`�n7@�������:              �?        a���       b�D�	>
����A�*�
w
discriminator_loss*a	   @�m>   @�m>      �?!   @�m>) 9����<2nx6�X� >�`��>�������:              �?        
s
generator_loss*a	    v56@    v56@      �?!    v56@) @&!��~@2}w�˝M5@�i*`�n7@�������:              �?        [r���       b�D�	C������A�*�
w
discriminator_loss*a	   `�;�=   `�;�=      �?!   `�;�=)@�L	W�;2�K���=�9�e��=�������:              �?        
s
generator_loss*a	   ��D6@   ��D6@      �?!   ��D6@)@��n�~@2}w�˝M5@�i*`�n7@�������:              �?        �r2	�       b�D�	]V�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �H?2#<2RT��+�>���">�������:              �?        
s
generator_loss*a	   ��J6@   ��J6@      �?!   ��J6@) y緛@2}w�˝M5@�i*`�n7@�������:              �?        �^$�       b�D�	��L���A�*�
w
discriminator_loss*a	   ��d>   ��d>      �?!   ��d>) ��1I<2��f��p>�i
�k>�������:              �?        
s
generator_loss*a	   @X6@   @X6@      �?!   @X6@) ��b4@2}w�˝M5@�i*`�n7@�������:              �?         �Q��       b�D�	�����A�*�
w
discriminator_loss*a	   `y'>   `y'>      �?!   `y'>) _b��`<2��o�kJ%>4��evk'>�������:              �?        
s
generator_loss*a	   ��q6@   ��q6@      �?!   ��q6@) i��_|@2}w�˝M5@�i*`�n7@�������:              �?        K���       b�D�	�B�	���A�*�
w
discriminator_loss*a	   �Q�=>   �Q�=>      �?!   �Q�=>)��0Ϊ�<2p
T~�;>����W_>>�������:              �?        
s
generator_loss*a	    A�6@    A�6@      �?!    A�6@) (醧@2}w�˝M5@�i*`�n7@�������:              �?        _� ��       b�D�	:����A�*�
w
discriminator_loss*a	   �֡>   �֡>      �?!   �֡>) I�pK<2�i
�k>%���>�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@) ���^�@2}w�˝M5@�i*`�n7@�������:              �?        0-Z�       b�D�	yi���A�*�
w
discriminator_loss*a	    O>    O>      �?!    O>) ���=NG<2��f��p>�i
�k>�������:              �?        
s
generator_loss*a	   �M�6@   �M�6@      �?!   �M�6@) d���@2}w�˝M5@�i*`�n7@�������:              �?        ��� �       b�D�	M�/���A�*�
w
discriminator_loss*a	   `�DS>   `�DS>      �?!   `�DS>)@rď�4�<2�
L�v�Q>H��'ϱS>�������:              �?        
s
generator_loss*a	   �b�6@   �b�6@      �?!   �b�6@)@�ޜ�@2}w�˝M5@�i*`�n7@�������:              �?        ��P�       b�D�	Ŕ����A�*�
w
discriminator_loss*a	   �<+>   �<+>      �?!   �<+>)@��l9<22!K�R�>��R���>�������:              �?        
s
generator_loss*a	    �j6@    �j6@      �?!    �j6@) ���h@2}w�˝M5@�i*`�n7@�������:              �?        }���       b�D�	�����A�*�
w
discriminator_loss*a	   @�i�=   @�i�=      �?!   @�i�=)�(_�	<2�tO���=�f׽r��=�������:              �?        
s
generator_loss*a	    �e6@    �e6@      �?!    �e6@)  ��Y@2}w�˝M5@�i*`�n7@�������:              �?        G'U/�       b�D�	� ���A�*�
w
discriminator_loss*a	   �?��=   �?��=      �?!   �?��=) q7��;2��-��J�=�K���=�������:              �?        
s
generator_loss*a	   @�|6@   @�|6@      �?!   @�|6@) %j��@2}w�˝M5@�i*`�n7@�������:              �?        �<�       b�D�	$#P$���A�*�
w
discriminator_loss*a	   ��0>   ��0>      �?!   ��0>) I�,p<27'_��+/>_"s�$1>�������:              �?        
s
generator_loss*a	   @r�6@   @r�6@      �?!   @r�6@) �wu�@2}w�˝M5@�i*`�n7@�������:              �?        .��B�       b�D�	�]%(���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) �3���&<2RT��+�>���">�������:              �?        
s
generator_loss*a	   ���6@   ���6@      �?!   ���6@) !!���@2}w�˝M5@�i*`�n7@�������:              �?        a^ ��       b�D�	��+���A�*�
w
discriminator_loss*a	   �oC>   �oC>      �?!   �oC>)@��I��<2�`��>�mm7&c>�������:              �?        
s
generator_loss*a	    f�6@    f�6@      �?!    f�6@)@؏%�@2}w�˝M5@�i*`�n7@�������:              �?        ����       b�D�	W�/���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) w�5��I<2��f��p>�i
�k>�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@) 0P�g �@2}w�˝M5@�i*`�n7@�������:              �?        ��k��       b�D�	J@�3���A�*�
w
discriminator_loss*a	   @��T>   @��T>      �?!   @��T>) q�=��<2H��'ϱS>��x��U>�������:              �?        
s
generator_loss*a	   ���6@   ���6@      �?!   ���6@) �ְ'�@2}w�˝M5@�i*`�n7@�������:              �?        �7���       b�D�	�Fu7���A�*�
w
discriminator_loss*a	     �[>     �[>      �?!     �[>)    Q��<24�j�6Z>��u}��\>�������:              �?        
s
generator_loss*a	   ���6@   ���6@      �?!   ���6@) �&X�;�@2}w�˝M5@�i*`�n7@�������:              �?        �K�       �N�	��3;���A*�
w
discriminator_loss*a	   @>   @>      �?!   @>)�h�-��(<2���">Z�TA[�>�������:              �?        
s
generator_loss*a	    ;�6@    ;�6@      �?!    ;�6@) �����@2}w�˝M5@�i*`�n7@�������:              �?        �ԑ�       �{�	�?���A(*�
w
discriminator_loss*a	    � >    � >      �?!    � >)@s
<2�f׽r��=nx6�X� >�������:              �?        
s
generator_loss*a	   ���6@   ���6@      �?!   ���6@)��c3�@2}w�˝M5@�i*`�n7@�������:              �?        (����       �{�	DY�B���AP*�
w
discriminator_loss*a	   ���`>   ���`>      �?!   ���`>)@�����<2d�V�_>w&���qa>�������:              �?        
s
generator_loss*a	   �P�6@   �P�6@      �?!   �P�6@) f���@2}w�˝M5@�i*`�n7@�������:              �?        �-P�       �{�	���F���Ax*�
w
discriminator_loss*a	   �"=>   �"=>      �?!   �"=>) yCqӒ2<2�#���j>�J>�������:              �?        
s
generator_loss*a	    ��6@    ��6@      �?!    ��6@) z?���@2}w�˝M5@�i*`�n7@�������:              �?        ���G�       b�D�	�5�J���A�*�
w
discriminator_loss*a	   �7>   �7>      �?!   �7>) �J^J��<2�z��6>u 5�9>�������:              �?        
s
generator_loss*a	    ٽ6@    ٽ6@      �?!    ٽ6@) :��p)�@2}w�˝M5@�i*`�n7@�������:              �?        C]�k�       b�D�	"�N���A�*�
w
discriminator_loss*a	   �՚>   �՚>      �?!   �՚>) t!��6/<2Z�TA[�>�#���j>�������:              �?        
s
generator_loss*a	   @��6@   @��6@      �?!   @��6@) �Hxl�@2}w�˝M5@�i*`�n7@�������:              �?        �<{��       b�D�	�^nR���A�*�
w
discriminator_loss*a	   @�rJ>   @�rJ>      �?!   @�rJ>)� U�ۥ<2��8"uH>6��>?�J>�������:              �?        
s
generator_loss*a	    �6@    �6@      �?!    �6@) $�6�@2}w�˝M5@�i*`�n7@�������:              �?        zp�       b�D�	��MV���A�*�
w
discriminator_loss*a	    ���=    ���=      �?!    ���=) �����;2��1���='j��p�=�������:              �?        
s
generator_loss*a	   @p�6@   @p�6@      �?!   @p�6@)�����.�@2}w�˝M5@�i*`�n7@�������:              �?        �ο��       b�D�	?vHZ���A�*�
w
discriminator_loss*a	   ���=   ���=      �?!   ���=) �tt��;2z�����=ݟ��uy�=�������:              �?        
s
generator_loss*a	   �O�6@   �O�6@      �?!   �O�6@) !��r?�@2}w�˝M5@�i*`�n7@�������:              �?        �+Ğ�       b�D�	1,^���A�*�
w
discriminator_loss*a	   ��K->   ��K->      �?!   ��K->)�pA.�j<2�'v�V,>7'_��+/>�������:              �?        
s
generator_loss*a	    }�6@    }�6@      �?!    }�6@) H���a�@2}w�˝M5@�i*`�n7@�������:              �?         ���       b�D�	
�b���A�*�
w
discriminator_loss*a	   @��5>   @��5>      �?!   @��5>) �m�%4~<2�so쩾4>�z��6>�������:              �?        
s
generator_loss*a	   �w�6@   �w�6@      �?!   �w�6@) B^��k�@2}w�˝M5@�i*`�n7@�������:              �?        �q���       b�D�	���e���A�*�
w
discriminator_loss*a	   ��dF>   ��dF>      �?!   ��dF>)@&!�W�<2��Ő�;F>��8"uH>�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@) �j�j�@2}w�˝M5@�i*`�n7@�������:              �?        �VG��       b�D�	4=�i���A�*�
w
discriminator_loss*a	   @�q>   @�q>      �?!   @�q>) �遡7<2�J>2!K�R�>�������:              �?        
s
generator_loss*a	   �X�6@   �X�6@      �?!   �X�6@) s�h�@2}w�˝M5@�i*`�n7@�������:              �?        ����       b�D�	�e�m���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)��a��"<2y�+pm>RT��+�>�������:              �?        
s
generator_loss*a	   �*�6@   �*�6@      �?!   �*�6@) ǎK͆�@2}w�˝M5@�i*`�n7@�������:              �?        ǆd�       b�D�	���q���A�*�
w
discriminator_loss*a	    V  >    V  >      �?!    V  >) @�Y�@P<2%���>��-�z�!>�������:              �?        
s
generator_loss*a	    W7@    W7@      �?!    W7@) ���,��@2}w�˝M5@�i*`�n7@�������:              �?        ����       b�D�	xQ�u���A�*�
w
discriminator_loss*a	   ��=   ��=      �?!   ��=)@�I����;2i@4[��=z�����=�������:              �?        
s
generator_loss*a	   �#7@   �#7@      �?!   �#7@) b�Tᴀ@2}w�˝M5@�i*`�n7@�������:              �?        d��<�       b�D�	��y���A�*�
w
discriminator_loss*a	   �-��=   �-��=      �?!   �-��=) �څ�]�;2ݟ��uy�=�/�4��=�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@) �p��b�@2}w�˝M5@�i*`�n7@�������:              �?        9�{�       b�D�	US�}���A�*�
w
discriminator_loss*a	   ���9>   ���9>      �?!   ���9>) �?�_�<2u 5�9>p
T~�;>�������:              �?        
s
generator_loss*a	   � �6@   � �6@      �?!   � �6@)�����b�@2}w�˝M5@�i*`�n7@�������:              �?        ٫���       b�D�	j�����A�*�
w
discriminator_loss*a	   ��z�=   ��z�=      �?!   ��z�=) �=��W�;2i@4[��=z�����=�������:              �?        
s
generator_loss*a	   ��6@   ��6@      �?!   ��6@)����Ux�@2}w�˝M5@�i*`�n7@�������:              �?        #����       b�D�	Q������A�*�
w
discriminator_loss*a	   ���	>   ���	>      �?!   ���	>) �b���$<2RT��+�>���">�������:              �?        
s
generator_loss*a	   `b�6@   `b�6@      �?!   `b�6@) m�~���@2}w�˝M5@�i*`�n7@�������:              �?        &�3�       b�D�	E㺉���A�*�
w
discriminator_loss*a	    e!�=    e!�=      �?!    e!�=) ��ѓ��;2H�����=PæҭU�=�������:              �?        
s
generator_loss*a	    �7@    �7@      �?!    �7@) ���:��@2}w�˝M5@�i*`�n7@�������:              �?        8���       b�D�	�\�����A�*�
w
discriminator_loss*a	    qv�=    qv�=      �?!    qv�=) �B��<2�tO���=�f׽r��=�������:              �?        
s
generator_loss*a	   @7@   @7@      �?!   @7@)��讀@2}w�˝M5@�i*`�n7@�������:              �?         ���       �N�	�暑���A*�
w
discriminator_loss*a	   �m�>   �m�>      �?!   �m�>)@�n��p<2�mm7&c>y�+pm>�������:              �?        
s
generator_loss*a	   @�7@   @�7@      �?!   @�7@)�$�:���@2}w�˝M5@�i*`�n7@�������:              �?        ��Y��       �{�	n{�����A(*�
w
discriminator_loss*a	   `��=   `��=      �?!   `��=)@>L��~�;2'j��p�=��-��J�=�������:              �?        
s
generator_loss*a	   ��7@   ��7@      �?!   ��7@) �Eᜀ@2}w�˝M5@�i*`�n7@�������:              �?        ��)"�       �{�	�������AP*�
w
discriminator_loss*a	    �$�=    �$�=      �?!    �$�=)@8T��^�;2�Qu�R"�=i@4[��=�������:              �?        
s
generator_loss*a	   ��7@   ��7@      �?!   ��7@)�h�r���@2}w�˝M5@�i*`�n7@�������:              �?        �(^��       �{�	�������Ax*�
w
discriminator_loss*a	    �
>    �
>      �?!    �
>) ��L&<2RT��+�>���">�������:              �?        
s
generator_loss*a	    �!7@    �!7@      �?!    �!7@) ��Ú��@2}w�˝M5@�i*`�n7@�������:              �?        ��Z�       b�D�	J������A�*�
w
discriminator_loss*a	    nX>    nX>      �?!    nX>) @���c7<2�J>2!K�R�>�������:              �?        
s
generator_loss*a	    �$7@    �$7@      �?!    �$7@) ȓ�J��@2}w�˝M5@�i*`�n7@�������:              �?        ��       b�D�	t�����A�*�
w
discriminator_loss*a	   �bJ�=   �bJ�=      �?!   �bJ�=) yѳP��;2�K���=�9�e��=�������:              �?        
s
generator_loss*a	    �<7@    �<7@      �?!    �<7@) ��؞߀@2}w�˝M5@�i*`�n7@�������:              �?        �����       b�D�	B�ǩ���A�*�
w
discriminator_loss*a	   ��0�=   ��0�=      �?!   ��0�=)@<P�>b�;2�Qu�R"�=i@4[��=�������:              �?        
s
generator_loss*a	    !K7@    !K7@      �?!    !K7@) Jh��@2}w�˝M5@�i*`�n7@�������:              �?        _M��       b�D�	�6ʭ���A�*�
w
discriminator_loss*a	   �{�>   �{�>      �?!   �{�>) D�]�3<2�#���j>�J>�������:              �?        
s
generator_loss*a	   ��J7@   ��J7@      �?!   ��J7@) ��,9�@2}w�˝M5@�i*`�n7@�������:              �?        �S �       b�D�	��ұ���A�*�
w
discriminator_loss*a	   @m��=   @m��=      �?!   @m��=)��8��?�;2��
"
�=���X>�=�������:              �?        
s
generator_loss*a	   �Y7@   �Y7@      �?!   �Y7@) �����@2}w�˝M5@�i*`�n7@�������:              �?        �<�       b�D�	z�۵���A�*�
w
discriminator_loss*a	   �6��=   �6��=      �?!   �6��=)���z�;2���X>�=H�����=�������:              �?        
s
generator_loss*a	   �3e7@   �3e7@      �?!   �3e7@)���r��@2}w�˝M5@�i*`�n7@�������:              �?        S%�w�       b�D�	�����A�*�
w
discriminator_loss*a	   @dF5>   @dF5>      �?!   @dF5>) !���I|<2�so쩾4>�z��6>�������:              �?        
s
generator_loss*a	   �Y|7@   �Y|7@      �?!   �Y|7@)���<�<�@2�i*`�n7@�6��9@�������:              �?        Xw��       b�D�	w������A�*�
w
discriminator_loss*a	   ���N>   ���N>      �?!   ���N>)��=�l5�<2������M>28���FP>�������:              �?        
s
generator_loss*a	    �w7@    �w7@      �?!    �w7@)  ���5�@2�i*`�n7@�6��9@�������:              �?        7K$��       b�D�	�4	����A�*�
w
discriminator_loss*a	   �j0>   �j0>      �?!   �j0>) ��a��><2��R���>Łt�=	>�������:              �?        
s
generator_loss*a	   �c7@   �c7@      �?!   �c7@) �.��@2}w�˝M5@�i*`�n7@�������:              �?        S�"�       b�D�	�����A�*�
w
discriminator_loss*a	    �s�=    �s�=      �?!    �s�=)@��Y$�;2�|86	�=��
"
�=�������:              �?        
s
generator_loss*a	   �3e7@   �3e7@      �?!   �3e7@)���r��@2}w�˝M5@�i*`�n7@�������:              �?        +tZ�       b�D�	��1����A�*�
w
discriminator_loss*a	    �7H>    �7H>      �?!    �7H>) �KKT�<2��Ő�;F>��8"uH>�������:              �?        
s
generator_loss*a	   ��l7@   ��l7@      �?!   ��l7@)����%�@2}w�˝M5@�i*`�n7@�������:              �?        �o�E�       b�D�	�WF����A�*�
w
discriminator_loss*a	   @�|->   @�|->      �?!   @�|->)���+k<2�'v�V,>7'_��+/>�������:              �?        
s
generator_loss*a	   `Fh7@   `Fh7@      �?!   `Fh7@) �
�8�@2}w�˝M5@�i*`�n7@�������:              �?        �W��       b�D�	�tm����A�*�
w
discriminator_loss*a	   ��\3>   ��\3>      �?!   ��\3>) �mnw<26NK��2>�so쩾4>�������:              �?        
s
generator_loss*a	   �7@   �7@      �?!   �7@) �O�@�@2�i*`�n7@�6��9@�������:              �?        ���T�       b�D�	�{�����A�*�
w
discriminator_loss*a	    {�
>    {�
>      �?!    {�
>) ����E&<2RT��+�>���">�������:              �?        
s
generator_loss*a	    �7@    �7@      �?!    �7@)  P B�@2�i*`�n7@�6��9@�������:              �?        ���q�       b�D�	������A�*�
w
discriminator_loss*a	   ����=   ����=      �?!   ����=) �.=i <2�9�e��=����%�=�������:              �?        
s
generator_loss*a	   ��v7@   ��v7@      �?!   ��v7@) &M4�@2�i*`�n7@�6��9@�������:              �?        j�	�       b�D�	�}�����A�*�
w
discriminator_loss*a	    ���=    ���=      �?!    ���=)@<.w���;2��-��J�=�K���=�������:              �?        
s
generator_loss*a	    �k7@    �k7@      �?!    �k7@) jZ
9$�@2}w�˝M5@�i*`�n7@�������:              �?        ����       b�D�	�������A�*�
w
discriminator_loss*a	    	�6>    	�6>      �?!    	�6>) �r�o�<2�z��6>u 5�9>�������:              �?        
s
generator_loss*a	    $w7@    $w7@      �?!    $w7@) �ւ�4�@2�i*`�n7@�6��9@�������:              �?        �"j�       b�D�	� �����A�*�
w
discriminator_loss*a	   @I��=   @I��=      �?!   @I��=)��s��p<2�tO���=�f׽r��=�������:              �?        
s
generator_loss*a	   @Gu7@   @Gu7@      �?!   @Gu7@)���=D2�@2�i*`�n7@�6��9@�������:              �?        V�fA�       �N�	B������A*�
w
discriminator_loss*a	    ���=    ���=      �?!    ���=) ���4��;2���X>�=H�����=�������:              �?        
s
generator_loss*a	   �b{7@   �b{7@      �?!   �b{7@) �]�9;�@2�i*`�n7@�6��9@�������:              �?        k+���       �{�	�*����A(*�
w
discriminator_loss*a	    �.�=    �.�=      �?!    �.�=) Jg�5F<2����%�=f;H�\Q�=�������:              �?        
s
generator_loss*a	   ��a7@   ��a7@      �?!   ��a7@) ����@2}w�˝M5@�i*`�n7@�������:              �?        ����       �{�	�?����AP*�
w
discriminator_loss*a	   @���=   @���=      �?!   @���=) � X�t�;2z�����=ݟ��uy�=�������:              �?        
s
generator_loss*a	    �q7@    �q7@      �?!    �q7@)  b$-�@2�i*`�n7@�6��9@�������:              �?        �����       �{�	!je����Ax*�
w
discriminator_loss*a	   @:/3>   @:/3>      �?!   @:/3>) �� w<26NK��2>�so쩾4>�������:              �?        
s
generator_loss*a	   �ˊ7@   �ˊ7@      �?!   �ˊ7@) �~�Q�@2�i*`�n7@�6��9@�������:              �?        �z��       b�D�	�ć����A�*�
w
discriminator_loss*a	    ?�j>    ?�j>      �?!    ?�j>) �V�>�<2ڿ�ɓ�i>=�.^ol>�������:              �?        
s
generator_loss*a	    ��7@    ��7@      �?!    ��7@) �jzdf�@2�i*`�n7@�6��9@�������:              �?        �ܘ�       b�D�	�������A�*�
w
discriminator_loss*a	   �!>   �!>      �?!   �!>)��w
�C<2Łt�=	>��f��p>�������:              �?        
s
generator_loss*a	   ��7@   ��7@      �?!   ��7@) 0�V\�@2�i*`�n7@�6��9@�������:              �?        n�2��       b�D�	O����A�*�
w
discriminator_loss*a	   �~;�=   �~;�=      �?!   �~;�=) ��y��;2ݟ��uy�=�/�4��=�������:              �?        
s
generator_loss*a	   `�7@   `�7@      �?!   `�7@) G�ۭu�@2�i*`�n7@�6��9@�������:              �?        �_R��       b�D�	Bz���A�*�
w
discriminator_loss*a	   ��M�=   ��M�=      �?!   ��M�=)���_��;2��.4N�=;3����=�������:              �?        
s
generator_loss*a	   �k�7@   �k�7@      �?!   �k�7@)���=Ɠ�@2�i*`�n7@�6��9@�������:              �?        	���       b�D�	��/���A�*�
w
discriminator_loss*a	    b�=    b�=      �?!    b�=) @Xw�m�;2i@4[��=z�����=�������:              �?        
s
generator_loss*a	   �w�7@   �w�7@      �?!   �w�7@) 1�5ƙ�@2�i*`�n7@�6��9@�������:              �?        ~e��       b�D�	��[���A�*�
w
discriminator_loss*a	    �3�=    �3�=      �?!    �3�=)@���;2i@4[��=z�����=�������:              �?        
s
generator_loss*a	    ��7@    ��7@      �?!    ��7@) 
�����@2�i*`�n7@�6��9@�������:              �?        ��t�       b�D�	�}����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) @���<2nx6�X� >�`��>�������:              �?        
s
generator_loss*a	   `��7@   `��7@      �?!   `��7@) �nժ��@2�i*`�n7@�6��9@�������:              �?        C3���       b�D�	�?����A�*�
w
discriminator_loss*a	   �0��=   �0��=      �?!   �0��=) �B�֋�;2�/�4��==��]���=�������:              �?        
s
generator_loss*a	   �H�7@   �H�7@      �?!   �H�7@) �Û��@2�i*`�n7@�6��9@�������:              �?        �8�	�       b�D�	�����A�*�
w
discriminator_loss*a	   ��7>   ��7>      �?!   ��7>) 9W 9� <2y�+pm>RT��+�>�������:              �?        
s
generator_loss*a	   `�7@   `�7@      �?!   `�7@) �
�
��@2�i*`�n7@�6��9@�������:              �?        �`f�       b�D�	�](!���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) ��SKIC<2Łt�=	>��f��p>�������:              �?        
s
generator_loss*a	   �h�7@   �h�7@      �?!   �h�7@) B�f��@2�i*`�n7@�6��9@�������:              �?        ,�M�       b�D�	�`%���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) �։�'<2���">Z�TA[�>�������:              �?        
s
generator_loss*a	   @x�7@   @x�7@      �?!   @x�7@)��F���@2�i*`�n7@�6��9@�������:              �?        ?@�z�       b�D�	3�)���A�*�
w
discriminator_loss*a	   �Y�=   �Y�=      �?!   �Y�=) �T���;2;3����=(�+y�6�=�������:              �?        
s
generator_loss*a	   @��7@   @��7@      �?!   @��7@)���Iׁ@2�i*`�n7@�6��9@�������:              �?        kӢe�       b�D�	�?�-���A�*�
w
discriminator_loss*a	   �m!�=   �m!�=      �?!   �m!�=) �N+G2�;2��
"
�=���X>�=�������:              �?        
s
generator_loss*a	   ���7@   ���7@      �?!   ���7@) �Ɇ�@2�i*`�n7@�6��9@�������:              �?        ��e��       b�D�	3�2���A�*�
w
discriminator_loss*a	   �k��=   �k��=      �?!   �k��=)@���#�;2��-��J�=�K���=�������:              �?        
s
generator_loss*a	   @68@   @68@      �?!   @68@)��#�Q�@2�i*`�n7@�6��9@�������:              �?        �!�0�       b�D�	��J6���A�*�
w
discriminator_loss*a	   ���=   ���=      �?!   ���=)��b#�'�;2���X>�=H�����=�������:              �?        
s
generator_loss*a	   ��7@   ��7@      �?!   ��7@) BH���@2�i*`�n7@�6��9@�������:              �?        �)U��       b�D�	�O�:���A�*�
w
discriminator_loss*a	    .�=    .�=      �?!    .�=)  ���I�;2��1���='j��p�=�������:              �?        
s
generator_loss*a	   �m
8@   �m
8@      �?!   �m
8@) ��է�@2�i*`�n7@�6��9@�������:              �?        ���       b�D�	�p�>���A�*�
w
discriminator_loss*a	   @���=   @���=      �?!   @���=)������;2PæҭU�=�Qu�R"�=�������:              �?        
s
generator_loss*a	   `
8@   `
8@      �?!   `
8@) 1>��@2�i*`�n7@�6��9@�������:              �?        �G��       b�D�	7�C���A�*�
w
discriminator_loss*a	    ���=    ���=      �?!    ���=)@d_��R�;2�Qu�R"�=i@4[��=�������:              �?        
s
generator_loss*a	    3�7@    3�7@      �?!    3�7@) �����@2�i*`�n7@�6��9@�������:              �?        I����       �N�	�j?G���A*�
w
discriminator_loss*a	   �6��=   �6��=      �?!   �6��=)@~�]N��;2ݟ��uy�=�/�4��=�������:              �?        
s
generator_loss*a	   ���7@   ���7@      �?!   ���7@) BK�N�@2�i*`�n7@�6��9@�������:              �?        �ߣ�       �{�	%��K���A(*�
w
discriminator_loss*a	   `��=   `��=      �?!   `��=) ����;2��.4N�=;3����=�������:              �?        
s
generator_loss*a	   ���7@   ���7@      �?!   ���7@) �����@2�i*`�n7@�6��9@�������:              �?        ���       �{�	���O���AP*�
w
discriminator_loss*a	   ���=   ���=      �?!   ���=) "|!Q��;2�!p/�^�=��.4N�=�������:              �?        
s
generator_loss*a	   @��7@   @��7@      �?!   @��7@)�`H�d��@2�i*`�n7@�6��9@�������:              �?        ����       �{�	��+T���Ax*�
w
discriminator_loss*a	    z��=    z��=      �?!    z��=) @��〵;2(�+y�6�=�|86	�=�������:              �?        
s
generator_loss*a	    8@    8@      �?!    8@) �
 �@2�i*`�n7@�6��9@�������:              �?        �ې�       b�D�	�sX���A�*�
w
discriminator_loss*a	    m�=    m�=      �?!    m�=)@D�ʣ+�;2�K���=�9�e��=�������:              �?        
s
generator_loss*a	   `�8@   `�8@      �?!   `�8@) I�\��@2�i*`�n7@�6��9@�������:              �?         :�&�       b�D�	��\���A�*�
w
discriminator_loss*a	   ���=   ���=      �?!   ���=) �1�8λ;2�|86	�=��
"
�=�������:              �?        
s
generator_loss*a	   ���7@   ���7@      �?!   ���7@) ����؁@2�i*`�n7@�6��9@�������:              �?        `���       b�D�	�a���A�*�
w
discriminator_loss*a	    �=    �=      �?!    �=) N��0�;2��.4N�=;3����=�������:              �?        
s
generator_loss*a	   @v�7@   @v�7@      �?!   @v�7@)��h�4��@2�i*`�n7@�6��9@�������:              �?        Q�_��       b�D�	�gWe���A�*�
w
discriminator_loss*a	    �=    �=      �?!    �=) ��f�@<2����%�=f;H�\Q�=�������:              �?        
s
generator_loss*a	   @M8@   @M8@      �?!   @M8@)�|Zv�@2�i*`�n7@�6��9@�������:              �?        �(���       b�D�	)�i���A�*�
w
discriminator_loss*a	   �]>   �]>      �?!   �]>)����(<2���">Z�TA[�>�������:              �?        
s
generator_loss*a	   ��
8@   ��
8@      �?!   ��
8@) q����@2�i*`�n7@�6��9@�������:              �?        w���       b�D�	�N�m���A�*�
w
discriminator_loss*a	   �ӽ�=   �ӽ�=      �?!   �ӽ�=) a��[�;2z�����=ݟ��uy�=�������:              �?        
s
generator_loss*a	   �Y	8@   �Y	8@      �?!   �Y	8@) RR��@2�i*`�n7@�6��9@�������:              �?        �
�       b�D�	6`Nr���A�*�
w
discriminator_loss*a	   `3��=   `3��=      �?!   `3��=)@��j�W�;2i@4[��=z�����=�������:              �?        
s
generator_loss*a	   ��"8@   ��"8@      �?!   ��"8@) 2��4�@2�i*`�n7@�6��9@�������:              �?        ֯��       b�D�	�)�v���A�*�
w
discriminator_loss*a	    <��=    <��=      �?!    <��=) ��7ѣ�;2�/�4��==��]���=�������:              �?        
s
generator_loss*a	   �58@   �58@      �?!   �58@) �1Ni*�@2�i*`�n7@�6��9@�������:              �?        hA���       b�D�	���z���A�*�
w
discriminator_loss*a	   �`�=   �`�=      �?!   �`�=) �%@��;2���X>�=H�����=�������:              �?        
s
generator_loss*a	   @�8@   @�8@      �?!   @�8@)�N]�@2�i*`�n7@�6��9@�������:              �?        #��O�       b�D�	��]���A�*�
w
discriminator_loss*a	   @"�=   @"�=      �?!   @"�=)��<�D�;2�d7����=�!p/�^�=�������:              �?        
s
generator_loss*a	   �8@   �8@      �?!   �8@)  ���-�@2�i*`�n7@�6��9@�������:              �?        1ai�       b�D�	�y�����A�*�
w
discriminator_loss*a	   `�g�=   `�g�=      �?!   `�g�=)@��,�;2i@4[��=z�����=�������:              �?        
s
generator_loss*a	   ��7@   ��7@      �?!   ��7@) �����@2}w�˝M5@�i*`�n7@�������:              �?        <)�Z�       b�D�	D&2����A�*�
w
discriminator_loss*a	    {��=    {��=      �?!    {��=) �Q��#�;2�Qu�R"�=i@4[��=�������:              �?        
s
generator_loss*a	   `��6@   `��6@      �?!   `��6@) ��8[�@2}w�˝M5@�i*`�n7@�������:              �?        "&�       b�D�	������A�*�
w
discriminator_loss*a	   �.^�=   �.^�=      �?!   �.^�=)�L�����;2PæҭU�=�Qu�R"�=�������:              �?        
s
generator_loss*a	   ��A7@   ��A7@      �?!   ��A7@)���Y&�@2}w�˝M5@�i*`�n7@�������:              �?        +��A�       b�D�	�������A�*�
w
discriminator_loss*a	   ���=   ���=      �?!   ���=) y� �޺;2�|86	�=��
"
�=�������:              �?        
s
generator_loss*a	    k�7@    k�7@      �?!    k�7@) ��A�I�@2�i*`�n7@�6��9@�������:              �?        �i���       b�D�	Op]����A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>)@�K�R	><2��R���>Łt�=	>�������:              �?        
s
generator_loss*a	    ��7@    ��7@      �?!    ��7@) �鷦��@2�i*`�n7@�6��9@�������:              �?        s��       b�D�	ę���A�*�
w
discriminator_loss*a	   �)�=   �)�=      �?!   �)�=) ��"�8�;2;3����=(�+y�6�=�������:              �?        
s
generator_loss*a	   ���7@   ���7@      �?!   ���7@) �+��@2�i*`�n7@�6��9@�������:              �?        (O�z�       b�D�	ɝ'����A�*�
w
discriminator_loss*a	   �*��=   �*��=      �?!   �*��=) ǜ{�ݭ;2��.4N�=;3����=�������:              �?        
s
generator_loss*a	   ���7@   ���7@      �?!   ���7@) �X�H�@2�i*`�n7@�6��9@�������:              �?        �'��       b�D�	�������A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@�8��!<2�mm7&c>y�+pm>�������:              �?        
s
generator_loss*a	   ��8@   ��8@      �?!   ��8@) �'�@2�i*`�n7@�6��9@�������:              �?        �h9�       �N�	lR٦���A*�
w
discriminator_loss*a	   @��=   @��=      �?!   @��=)��LU�I�;2��.4N�=;3����=�������:              �?        
s
generator_loss*a	   ��$8@   ��$8@      �?!   ��$8@) RM�7�@2�i*`�n7@�6��9@�������:              �?        ����       �{�	��H����A(*�
w
discriminator_loss*a	   `ܶ�=   `ܶ�=      �?!   `ܶ�=) ��Fͩ<2f;H�\Q�=�tO���=�������:              �?        
s
generator_loss*a	   ��<8@   ��<8@      �?!   ��<8@) X3}P[�@2�i*`�n7@�6��9@�������:              �?        C2%n�       �{�	rf�����AP*�
w
discriminator_loss*a	   �U��=   �U��=      �?!   �U��=) �(�nl�;2;3����=(�+y�6�=�������:              �?        
s
generator_loss*a	   ��=8@   ��=8@      �?!   ��=8@)�eK]�@2�i*`�n7@�6��9@�������:              �?        *�&	�       �{�	��#����Ax*�
w
discriminator_loss*a	   �H�=   �H�=      �?!   �H�=) .���l�;2���X>�=H�����=�������:              �?        
s
generator_loss*a	    I?8@    I?8@      �?!    I?8@) ��j_�@2�i*`�n7@�6��9@�������:              �?        �&4��       b�D�	'{�����A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>)@\f�ؼ<2�mm7&c>y�+pm>�������:              �?        
s
generator_loss*a	   @�N8@   @�N8@      �?!   @�N8@)�tlF�v�@2�i*`�n7@�6��9@�������:              �?         ޓ�       b�D�	������A�*�
w
discriminator_loss*a	   �9Y>   �9Y>      �?!   �9Y>)@���9<22!K�R�>��R���>�������:              �?        
s
generator_loss*a	   �Ea8@   �Ea8@      �?!   �Ea8@) }���@2�i*`�n7@�6��9@�������:              �?        FT�       b�D�	�������A�*�
w
discriminator_loss*a	   `�ѽ=   `�ѽ=      �?!   `�ѽ=) !��mɋ;25%���=�Bb�!�=�������:              �?        
s
generator_loss*a	   @�[8@   @�[8@      �?!   @�[8@)�(�I��@2�i*`�n7@�6��9@�������:              �?        f?�'�       b�D�	�����A�*�
w
discriminator_loss*a	   @�$�=   @�$�=      �?!   @�$�=)� u�V7�;2��
"
�=���X>�=�������:              �?        
s
generator_loss*a	   @kn8@   @kn8@      �?!   @kn8@)�t����@2�i*`�n7@�6��9@�������:              �?        ����       b�D�	�Ճ����A�*�
w
discriminator_loss*a	   @a�1>   @a�1>      �?!   @a�1>) 'Q��s<2_"s�$1>6NK��2>�������:              �?        
s
generator_loss*a	   `x8@   `x8@      �?!   `x8@) �P8ʵ�@2�i*`�n7@�6��9@�������:              �?        ����       b�D�	�9�����A�*�
w
discriminator_loss*a	    ���=    ���=      �?!    ���=) ��$��;2��1���='j��p�=�������:              �?        
s
generator_loss*a	   �x8@   �x8@      �?!   �x8@) <����@2�i*`�n7@�6��9@�������:              �?        ɦ�w�       b�D�	{�v����A�*�
w
discriminator_loss*a	    ��=    ��=      �?!    ��=)  ���;2�|86	�=��
"
�=�������:              �?        
s
generator_loss*a	   `z8@   `z8@      �?!   `z8@) ����@2�i*`�n7@�6��9@�������:              �?        ��q?�       b�D�	E������A�*�
w
discriminator_loss*a	    I<)>    I<)>      �?!    I<)>) �ԣ�c<24��evk'>���<�)>�������:              �?        
s
generator_loss*a	   �p8@   �p8@      �?!   �p8@) B d���@2�i*`�n7@�6��9@�������:              �?        �lL_�       b�D�	�z����A�*�
w
discriminator_loss*a	   ��"�=   ��"�=      �?!   ��"�=) �>uퟞ;2�b1��=��؜��=�������:              �?        
s
generator_loss*a	   `�|8@   `�|8@      �?!   `�|8@) �B��@2�i*`�n7@�6��9@�������:              �?        Fs��       b�D�	�������A�*�
w
discriminator_loss*a	    �}�=    �}�=      �?!    �}�=) ���-�;25%���=�Bb�!�=�������:              �?        
s
generator_loss*a	   `J�8@   `J�8@      �?!   `J�8@) ����߂@2�i*`�n7@�6��9@�������:              �?        �����       b�D�	#f~����A�*�
w
discriminator_loss*a	    Ek>    Ek>      �?!    Ek>)@��Y��7<2�J>2!K�R�>�������:              �?        
s
generator_loss*a	   `��8@   `��8@      �?!   `��8@) )�����@2�i*`�n7@�6��9@�������:              �?        ,�j�       b�D�	������A�*�
w
discriminator_loss*a	   �xJ�=   �xJ�=      �?!   �xJ�=)��?,�ϊ;25%���=�Bb�!�=�������:              �?        
s
generator_loss*a	   `��8@   `��8@      �?!   `��8@) +犢�@2�i*`�n7@�6��9@�������:              �?        [���       b�D�	������A�*�
w
discriminator_loss*a	   @��=   @��=      �?!   @��=)����(�;2PæҭU�=�Qu�R"�=�������:              �?        
s
generator_loss*a	   @��8@   @��8@      �?!   @��8@)��އ��@2�i*`�n7@�6��9@�������:              �?        �f��       b�D�	T~����A�*�
w
discriminator_loss*a	    U>    U>      �?!    U>)  �d�0<2Z�TA[�>�#���j>�������:              �?        
s
generator_loss*a	   @w�8@   @w�8@      �?!   @w�8@)�d��ق@2�i*`�n7@�6��9@�������:              �?        ��c��       b�D�	΁�����A�*�
w
discriminator_loss*a	   @X�=   @X�=      �?!   @X�=)� b����;2�d7����=�!p/�^�=�������:              �?        
s
generator_loss*a	    �i8@    �i8@      �?!    �i8@) i/E��@2�i*`�n7@�6��9@�������:              �?        �\�J�       b�D�	��E����A�*�
w
discriminator_loss*a	   @=?�=   @=?�=      �?!   @=?�=) yB��;2K?�\���=�b1��=�������:              �?        
s
generator_loss*a	   `�p8@   `�p8@      �?!   `�p8@) ⊁��@2�i*`�n7@�6��9@�������:              �?        H����       b�D�	ٻ� ���A�*�
w
discriminator_loss*a	   `}4�=   `}4�=      �?!   `}4�=)@n��i�;2��.4N�=;3����=�������:              �?        
s
generator_loss*a	   �H�8@   �H�8@      �?!   �H�8@) ���т@2�i*`�n7@�6��9@�������:              �?        �K���       b�D�	+�h���A�*�
w
discriminator_loss*a	    �E�=    �E�=      �?!    �E�=) ��� �;2��
"
�=���X>�=�������:              �?        
s
generator_loss*a	   �;�8@   �;�8@      �?!   �;�8@) jz��@2�i*`�n7@�6��9@�������:              �?        ZT�       �N�	�+�	���A*�
w
discriminator_loss*a	   ��%>   ��%>      �?!   ��%>) �"˾@<2��R���>Łt�=	>�������:              �?        
s
generator_loss*a	    ��8@    ��8@      �?!    ��8@)  �1�@2�i*`�n7@�6��9@�������:              �?        SJ��       �{�	I't���A(*�
w
discriminator_loss*a	   �D?�=   �D?�=      �?!   �D?�=)@V�l��;2�Qu�R"�=i@4[��=�������:              �?        
s
generator_loss*a	   ���8@   ���8@      �?!   ���8@) ���j�@2�i*`�n7@�6��9@�������:              �?        ��̬�       �{�	����AP*�
w
discriminator_loss*a	   `��=   `��=      �?!   `��=) �9��;2�/�4��==��]���=�������:              �?        
s
generator_loss*a	   @u�8@   @u�8@      �?!   @u�8@)����K�@2�i*`�n7@�6��9@�������:              �?        >1���       �{�	7����Ax*�
w
discriminator_loss*a	   ����=   ����=      �?!   ����=) p�[�;2;3����=(�+y�6�=�������:              �?        
s
generator_loss*a	   ���8@   ���8@      �?!   ���8@) U�G��@2�i*`�n7@�6��9@�������:              �?        ܡ?p�       b�D�	�|6���A�*�
w
discriminator_loss*a	    \��=    \��=      �?!    \��=) 8_iӂ;2���6�=G�L��=�������:              �?        
s
generator_loss*a	   ���8@   ���8@      �?!   ���8@) ��20�@2�i*`�n7@�6��9@�������:              �?        
a�v�       b�D�	��� ���A�*�
w
discriminator_loss*a	   ��=   ��=      �?!   ��=) ����;2���X>�=H�����=�������:              �?        
s
generator_loss*a	   ��8@   ��8@      �?!   ��8@) 	p��L�@2�i*`�n7@�6��9@�������:              �?        �Ϛ�