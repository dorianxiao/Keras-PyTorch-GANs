       �K"	  �^���Abrain.Event:2�C\k��     �;�	��^���A"��
r
Generator/noisePlaceholder*
dtype0*'
_output_shapes
:���������d*
shape:���������d
�
VGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
�
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&�*
dtype0*
_output_shapes
: 
�
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&>*
dtype0*
_output_shapes
: 
�
^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d�*

seed*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
seed2
�
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
: 
�
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/sub*
_output_shapes
:	d�*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
�
PGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniformAddTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
:	d�*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
�
5Generator/first/Generator/firstfully_connected/kernel
VariableV2*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�
�
<Generator/first/Generator/firstfully_connected/kernel/AssignAssign5Generator/first/Generator/firstfully_connected/kernelPGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0
�
:Generator/first/Generator/firstfully_connected/kernel/readIdentity5Generator/first/Generator/firstfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�
�
EGenerator/first/Generator/firstfully_connected/bias/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3Generator/first/Generator/firstfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:�
�
:Generator/first/Generator/firstfully_connected/bias/AssignAssign3Generator/first/Generator/firstfully_connected/biasEGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
8Generator/first/Generator/firstfully_connected/bias/readIdentity3Generator/first/Generator/firstfully_connected/bias*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:�
�
5Generator/first/Generator/firstfully_connected/MatMulMatMulGenerator/noise:Generator/first/Generator/firstfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
6Generator/first/Generator/firstfully_connected/BiasAddBiasAdd5Generator/first/Generator/firstfully_connected/MatMul8Generator/first/Generator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
t
/Generator/first/Generator/firstleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
-Generator/first/Generator/firstleaky_relu/mulMul/Generator/first/Generator/firstleaky_relu/alpha6Generator/first/Generator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
)Generator/first/Generator/firstleaky_reluMaximum-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
XGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   �
�
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
�
`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformXGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shape*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
seed2*
dtype0* 
_output_shapes
:
��*

seed
�
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/subSubVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
_output_shapes
: 
�
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulMul`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
�
RGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniformAddVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
�
7Generator/second/Generator/secondfully_connected/kernel
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container 
�
>Generator/second/Generator/secondfully_connected/kernel/AssignAssign7Generator/second/Generator/secondfully_connected/kernelRGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
<Generator/second/Generator/secondfully_connected/kernel/readIdentity7Generator/second/Generator/secondfully_connected/kernel*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
�
GGenerator/second/Generator/secondfully_connected/bias/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
5Generator/second/Generator/secondfully_connected/bias
VariableV2*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
<Generator/second/Generator/secondfully_connected/bias/AssignAssign5Generator/second/Generator/secondfully_connected/biasGGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(
�
:Generator/second/Generator/secondfully_connected/bias/readIdentity5Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
�
7Generator/second/Generator/secondfully_connected/MatMulMatMul)Generator/first/Generator/firstleaky_relu<Generator/second/Generator/secondfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
8Generator/second/Generator/secondfully_connected/BiasAddBiasAdd7Generator/second/Generator/secondfully_connected/MatMul:Generator/second/Generator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
HGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:�*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB�*  �?
�
7Generator/second/Generator/secondbatch_normalized/gamma
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:�
�
>Generator/second/Generator/secondbatch_normalized/gamma/AssignAssign7Generator/second/Generator/secondbatch_normalized/gammaHGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
<Generator/second/Generator/secondbatch_normalized/gamma/readIdentity7Generator/second/Generator/secondbatch_normalized/gamma*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:�
�
HGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
6Generator/second/Generator/secondbatch_normalized/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:�
�
=Generator/second/Generator/secondbatch_normalized/beta/AssignAssign6Generator/second/Generator/secondbatch_normalized/betaHGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
;Generator/second/Generator/secondbatch_normalized/beta/readIdentity6Generator/second/Generator/secondbatch_normalized/beta*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:�
�
OGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zerosConst*
_output_shapes	
:�*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
valueB�*    *
dtype0
�
=Generator/second/Generator/secondbatch_normalized/moving_mean
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
	container *
shape:�
�
DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignAssign=Generator/second/Generator/secondbatch_normalized/moving_meanOGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
BGenerator/second/Generator/secondbatch_normalized/moving_mean/readIdentity=Generator/second/Generator/secondbatch_normalized/moving_mean*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
_output_shapes	
:�
�
RGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:�*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
valueB�*  �?
�
AGenerator/second/Generator/secondbatch_normalized/moving_variance
VariableV2*
shared_name *T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
HGenerator/second/Generator/secondbatch_normalized/moving_variance/AssignAssignAGenerator/second/Generator/secondbatch_normalized/moving_varianceRGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/ones*
use_locking(*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:�
�
FGenerator/second/Generator/secondbatch_normalized/moving_variance/readIdentityAGenerator/second/Generator/secondbatch_normalized/moving_variance*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
_output_shapes	
:�
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
?Generator/second/Generator/secondbatch_normalized/batchnorm/addAddFGenerator/second/Generator/secondbatch_normalized/moving_variance/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:�
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/RsqrtRsqrt?Generator/second/Generator/secondbatch_normalized/batchnorm/add*
_output_shapes	
:�*
T0
�
?Generator/second/Generator/secondbatch_normalized/batchnorm/mulMulAGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt<Generator/second/Generator/secondbatch_normalized/gamma/read*
T0*
_output_shapes	
:�
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAdd?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*(
_output_shapes
:����������*
T0
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2MulBGenerator/second/Generator/secondbatch_normalized/moving_mean/read?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
_output_shapes	
:�*
T0
�
?Generator/second/Generator/secondbatch_normalized/batchnorm/subSub;Generator/second/Generator/secondbatch_normalized/beta/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1AddAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1?Generator/second/Generator/secondbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:����������
v
1Generator/second/Generator/secondleaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
�
/Generator/second/Generator/secondleaky_relu/mulMul1Generator/second/Generator/secondleaky_relu/alphaAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
+Generator/second/Generator/secondleaky_reluMaximum/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
VGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *��=*
dtype0*
_output_shapes
: 
�
^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
seed2B
�
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
_output_shapes
: 
�
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
�
PGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniformAddTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��*
T0
�
5Generator/third/Generator/thirdfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
��
�
<Generator/third/Generator/thirdfully_connected/kernel/AssignAssign5Generator/third/Generator/thirdfully_connected/kernelPGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
:Generator/third/Generator/thirdfully_connected/kernel/readIdentity5Generator/third/Generator/thirdfully_connected/kernel*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��*
T0
�
EGenerator/third/Generator/thirdfully_connected/bias/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3Generator/third/Generator/thirdfully_connected/bias
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
�
:Generator/third/Generator/thirdfully_connected/bias/AssignAssign3Generator/third/Generator/thirdfully_connected/biasEGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
8Generator/third/Generator/thirdfully_connected/bias/readIdentity3Generator/third/Generator/thirdfully_connected/bias*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:�*
T0
�
5Generator/third/Generator/thirdfully_connected/MatMulMatMul+Generator/second/Generator/secondleaky_relu:Generator/third/Generator/thirdfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
6Generator/third/Generator/thirdfully_connected/BiasAddBiasAdd5Generator/third/Generator/thirdfully_connected/MatMul8Generator/third/Generator/thirdfully_connected/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
�
FGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/onesConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
5Generator/third/Generator/thirdbatch_normalized/gamma
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container 
�
<Generator/third/Generator/thirdbatch_normalized/gamma/AssignAssign5Generator/third/Generator/thirdbatch_normalized/gammaFGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
:Generator/third/Generator/thirdbatch_normalized/gamma/readIdentity5Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
�
FGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
4Generator/third/Generator/thirdbatch_normalized/beta
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
�
;Generator/third/Generator/thirdbatch_normalized/beta/AssignAssign4Generator/third/Generator/thirdbatch_normalized/betaFGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(
�
9Generator/third/Generator/thirdbatch_normalized/beta/readIdentity4Generator/third/Generator/thirdbatch_normalized/beta*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:�*
T0
�
MGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zerosConst*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
;Generator/third/Generator/thirdbatch_normalized/moving_mean
VariableV2*
shared_name *N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
BGenerator/third/Generator/thirdbatch_normalized/moving_mean/AssignAssign;Generator/third/Generator/thirdbatch_normalized/moving_meanMGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:�
�
@Generator/third/Generator/thirdbatch_normalized/moving_mean/readIdentity;Generator/third/Generator/thirdbatch_normalized/moving_mean*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
_output_shapes	
:�
�
PGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/onesConst*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
?Generator/third/Generator/thirdbatch_normalized/moving_variance
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance
�
FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignAssign?Generator/third/Generator/thirdbatch_normalized/moving_variancePGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
DGenerator/third/Generator/thirdbatch_normalized/moving_variance/readIdentity?Generator/third/Generator/thirdbatch_normalized/moving_variance*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
_output_shapes	
:�
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
=Generator/third/Generator/thirdbatch_normalized/batchnorm/addAddDGenerator/third/Generator/thirdbatch_normalized/moving_variance/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:�
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/RsqrtRsqrt=Generator/third/Generator/thirdbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:�
�
=Generator/third/Generator/thirdbatch_normalized/batchnorm/mulMul?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:�
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAdd=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2Mul@Generator/third/Generator/thirdbatch_normalized/moving_mean/read=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:�
�
=Generator/third/Generator/thirdbatch_normalized/batchnorm/subSub9Generator/third/Generator/thirdbatch_normalized/beta/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1Add?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:����������
t
/Generator/third/Generator/thirdleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
-Generator/third/Generator/thirdleaky_relu/mulMul/Generator/third/Generator/thirdleaky_relu/alpha?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
)Generator/third/Generator/thirdleaky_reluMaximum-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
VGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0
�
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  �=*
dtype0*
_output_shapes
: 
�
^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
seed2m
�
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
_output_shapes
: 
�
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��
�
PGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniformAddTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��
�
5Generator/forth/Generator/forthfully_connected/kernel
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
�
<Generator/forth/Generator/forthfully_connected/kernel/AssignAssign5Generator/forth/Generator/forthfully_connected/kernelPGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
:Generator/forth/Generator/forthfully_connected/kernel/readIdentity5Generator/forth/Generator/forthfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��
�
UGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
KGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
EGenerator/forth/Generator/forthfully_connected/bias/Initializer/zerosFillUGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorKGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:�
�
3Generator/forth/Generator/forthfully_connected/bias
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container 
�
:Generator/forth/Generator/forthfully_connected/bias/AssignAssign3Generator/forth/Generator/forthfully_connected/biasEGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
8Generator/forth/Generator/forthfully_connected/bias/readIdentity3Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
�
5Generator/forth/Generator/forthfully_connected/MatMulMatMul)Generator/third/Generator/thirdleaky_relu:Generator/forth/Generator/forthfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
6Generator/forth/Generator/forthfully_connected/BiasAddBiasAdd5Generator/forth/Generator/forthfully_connected/MatMul8Generator/forth/Generator/forthfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
VGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:�*
dtype0*
_output_shapes
:
�
LGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
FGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/onesFillVGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:�
�
5Generator/forth/Generator/forthbatch_normalized/gamma
VariableV2*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
<Generator/forth/Generator/forthbatch_normalized/gamma/AssignAssign5Generator/forth/Generator/forthbatch_normalized/gammaFGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(
�
:Generator/forth/Generator/forthbatch_normalized/gamma/readIdentity5Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
�
VGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:�*
dtype0*
_output_shapes
:
�
LGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
FGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zerosFillVGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:�
�
4Generator/forth/Generator/forthbatch_normalized/beta
VariableV2*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
;Generator/forth/Generator/forthbatch_normalized/beta/AssignAssign4Generator/forth/Generator/forthbatch_normalized/betaFGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
9Generator/forth/Generator/forthbatch_normalized/beta/readIdentity4Generator/forth/Generator/forthbatch_normalized/beta*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:�
�
]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB:�*
dtype0
�
SGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/ConstConst*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
MGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/Const*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*

index_type0*
_output_shapes	
:�
�
;Generator/forth/Generator/forthbatch_normalized/moving_mean
VariableV2*
shared_name *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignAssign;Generator/forth/Generator/forthbatch_normalized/moving_meanMGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:�
�
@Generator/forth/Generator/forthbatch_normalized/moving_mean/readIdentity;Generator/forth/Generator/forthbatch_normalized/moving_mean*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
_output_shapes	
:�*
T0
�
`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB:�*
dtype0
�
VGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/ConstConst*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
PGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/onesFill`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorVGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/Const*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*

index_type0*
_output_shapes	
:�
�
?Generator/forth/Generator/forthbatch_normalized/moving_variance
VariableV2*
shared_name *R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignAssign?Generator/forth/Generator/forthbatch_normalized/moving_variancePGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones*
use_locking(*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:�
�
DGenerator/forth/Generator/forthbatch_normalized/moving_variance/readIdentity?Generator/forth/Generator/forthbatch_normalized/moving_variance*
_output_shapes	
:�*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
=Generator/forth/Generator/forthbatch_normalized/batchnorm/addAddDGenerator/forth/Generator/forthbatch_normalized/moving_variance/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/y*
_output_shapes	
:�*
T0
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/RsqrtRsqrt=Generator/forth/Generator/forthbatch_normalized/batchnorm/add*
_output_shapes	
:�*
T0
�
=Generator/forth/Generator/forthbatch_normalized/batchnorm/mulMul?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt:Generator/forth/Generator/forthbatch_normalized/gamma/read*
T0*
_output_shapes	
:�
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAdd=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*(
_output_shapes
:����������*
T0
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2Mul@Generator/forth/Generator/forthbatch_normalized/moving_mean/read=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
_output_shapes	
:�*
T0
�
=Generator/forth/Generator/forthbatch_normalized/batchnorm/subSub9Generator/forth/Generator/forthbatch_normalized/beta/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1Add?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:����������
t
/Generator/forth/Generator/forthleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
-Generator/forth/Generator/forthleaky_relu/mulMul/Generator/forth/Generator/forthleaky_relu/alpha?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
)Generator/forth/Generator/forthleaky_reluMaximum-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
7Generator/dense/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
5Generator/dense/kernel/Initializer/random_uniform/minConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *z�k�*
dtype0*
_output_shapes
: 
�
5Generator/dense/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *z�k=*
dtype0*
_output_shapes
: 
�
?Generator/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7Generator/dense/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed*
T0*)
_class
loc:@Generator/dense/kernel*
seed2�
�
5Generator/dense/kernel/Initializer/random_uniform/subSub5Generator/dense/kernel/Initializer/random_uniform/max5Generator/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@Generator/dense/kernel
�
5Generator/dense/kernel/Initializer/random_uniform/mulMul?Generator/dense/kernel/Initializer/random_uniform/RandomUniform5Generator/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
��
�
1Generator/dense/kernel/Initializer/random_uniformAdd5Generator/dense/kernel/Initializer/random_uniform/mul5Generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
��
�
Generator/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:
��
�
Generator/dense/kernel/AssignAssignGenerator/dense/kernel1Generator/dense/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel
�
Generator/dense/kernel/readIdentityGenerator/dense/kernel* 
_output_shapes
:
��*
T0*)
_class
loc:@Generator/dense/kernel
�
&Generator/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*'
_class
loc:@Generator/dense/bias*
valueB�*    
�
Generator/dense/bias
VariableV2*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
Generator/dense/bias/AssignAssignGenerator/dense/bias&Generator/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias
�
Generator/dense/bias/readIdentityGenerator/dense/bias*
_output_shapes	
:�*
T0*'
_class
loc:@Generator/dense/bias
�
Generator/dense/MatMulMatMul)Generator/forth/Generator/forthleaky_reluGenerator/dense/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
Generator/dense/BiasAddBiasAddGenerator/dense/MatMulGenerator/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
b
Generator/TanhTanhGenerator/dense/BiasAdd*
T0*(
_output_shapes
:����������
w
Discriminator/realPlaceholder*
dtype0*(
_output_shapes
:����������*
shape:����������
�
^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HY��*
dtype0
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HY�=*
dtype0*
_output_shapes
: 
�
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
seed2�
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/subSub\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/max\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mulMulfDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniform\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
�
XDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniformAdd\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mul\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
�
=Discriminator/first/Discriminator/firstfully_connected/kernel
VariableV2*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignAssign=Discriminator/first/Discriminator/firstfully_connected/kernelXDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
BDiscriminator/first/Discriminator/firstfully_connected/kernel/readIdentity=Discriminator/first/Discriminator/firstfully_connected/kernel*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
�
MDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
;Discriminator/first/Discriminator/firstfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:�
�
BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignAssign;Discriminator/first/Discriminator/firstfully_connected/biasMDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
@Discriminator/first/Discriminator/firstfully_connected/bias/readIdentity;Discriminator/first/Discriminator/firstfully_connected/bias*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:�
�
=Discriminator/first/Discriminator/firstfully_connected/MatMulMatMulDiscriminator/realBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
>Discriminator/first/Discriminator/firstfully_connected/BiasAddBiasAdd=Discriminator/first/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
|
7Discriminator/first/Discriminator/firstleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
5Discriminator/first/Discriminator/firstleaky_relu/mulMul7Discriminator/first/Discriminator/firstleaky_relu/alpha>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
1Discriminator/first/Discriminator/firstleaky_reluMaximum5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/minConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *��=*
dtype0*
_output_shapes
: 
�
hDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
seed2�
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/subSub^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/max^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mulMulhDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniform^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/sub*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��
�
ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniformAdd^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mul^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��*
T0
�
?Discriminator/second/Discriminator/secondfully_connected/kernel
VariableV2*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignAssign?Discriminator/second/Discriminator/secondfully_connected/kernelZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
�
DDiscriminator/second/Discriminator/secondfully_connected/kernel/readIdentity?Discriminator/second/Discriminator/secondfully_connected/kernel*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��
�
ODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
=Discriminator/second/Discriminator/secondfully_connected/bias
VariableV2*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignAssign=Discriminator/second/Discriminator/secondfully_connected/biasODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
BDiscriminator/second/Discriminator/secondfully_connected/bias/readIdentity=Discriminator/second/Discriminator/secondfully_connected/bias*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:�
�
?Discriminator/second/Discriminator/secondfully_connected/MatMulMatMul1Discriminator/first/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
@Discriminator/second/Discriminator/secondfully_connected/BiasAddBiasAdd?Discriminator/second/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
~
9Discriminator/second/Discriminator/secondleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
7Discriminator/second/Discriminator/secondleaky_relu/mulMul9Discriminator/second/Discriminator/secondleaky_relu/alpha@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
3Discriminator/second/Discriminator/secondleaky_reluMaximum7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
9Discriminator/out/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@Discriminator/out/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
7Discriminator/out/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv�*
dtype0*
_output_shapes
: 
�
7Discriminator/out/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
�
ADiscriminator/out/kernel/Initializer/random_uniform/RandomUniformRandomUniform9Discriminator/out/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@Discriminator/out/kernel*
seed2�*
dtype0*
_output_shapes
:	�*

seed
�
7Discriminator/out/kernel/Initializer/random_uniform/subSub7Discriminator/out/kernel/Initializer/random_uniform/max7Discriminator/out/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
: 
�
7Discriminator/out/kernel/Initializer/random_uniform/mulMulADiscriminator/out/kernel/Initializer/random_uniform/RandomUniform7Discriminator/out/kernel/Initializer/random_uniform/sub*
_output_shapes
:	�*
T0*+
_class!
loc:@Discriminator/out/kernel
�
3Discriminator/out/kernel/Initializer/random_uniformAdd7Discriminator/out/kernel/Initializer/random_uniform/mul7Discriminator/out/kernel/Initializer/random_uniform/min*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�*
T0
�
Discriminator/out/kernel
VariableV2*
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container 
�
Discriminator/out/kernel/AssignAssignDiscriminator/out/kernel3Discriminator/out/kernel/Initializer/random_uniform*
_output_shapes
:	�*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(
�
Discriminator/out/kernel/readIdentityDiscriminator/out/kernel*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�
�
(Discriminator/out/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*)
_class
loc:@Discriminator/out/bias*
valueB*    
�
Discriminator/out/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:
�
Discriminator/out/bias/AssignAssignDiscriminator/out/bias(Discriminator/out/bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(
�
Discriminator/out/bias/readIdentityDiscriminator/out/bias*
T0*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:
�
Discriminator/out/MatMulMatMul3Discriminator/second/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
Discriminator/out/BiasAddBiasAddDiscriminator/out/MatMulDiscriminator/out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
q
Discriminator/out/SigmoidSigmoidDiscriminator/out/BiasAdd*'
_output_shapes
:���������*
T0
�
?Discriminator/first_1/Discriminator/firstfully_connected/MatMulMatMulGenerator/TanhBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
@Discriminator/first_1/Discriminator/firstfully_connected/BiasAddBiasAdd?Discriminator/first_1/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
~
9Discriminator/first_1/Discriminator/firstleaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
�
7Discriminator/first_1/Discriminator/firstleaky_relu/mulMul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
3Discriminator/first_1/Discriminator/firstleaky_reluMaximum7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
ADiscriminator/second_1/Discriminator/secondfully_connected/MatMulMatMul3Discriminator/first_1/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
BDiscriminator/second_1/Discriminator/secondfully_connected/BiasAddBiasAddADiscriminator/second_1/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
�
;Discriminator/second_1/Discriminator/secondleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
9Discriminator/second_1/Discriminator/secondleaky_relu/mulMul;Discriminator/second_1/Discriminator/secondleaky_relu/alphaBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
5Discriminator/second_1/Discriminator/secondleaky_reluMaximum9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Discriminator/out_1/MatMulMatMul5Discriminator/second_1/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
Discriminator/out_1/BiasAddBiasAddDiscriminator/out_1/MatMulDiscriminator/out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
u
Discriminator/out_1/SigmoidSigmoidDiscriminator/out_1/BiasAdd*'
_output_shapes
:���������*
T0
W
LogLogDiscriminator/out/Sigmoid*
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
`
subSubsub/xDiscriminator/out_1/Sigmoid*'
_output_shapes
:���������*
T0
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
discriminator_lossHistogramSummarydiscriminator_loss/tagNeg*
_output_shapes
: *
T0
L
sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
d
sub_1Subsub_1/xDiscriminator/out_1/Sigmoid*
T0*'
_output_shapes
:���������
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
Mean_1MeanLog_2Const_1*
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
valueB"      *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
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
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
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
gradients/add_grad/SumSumgradients/Mean_grad/truediv(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
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
ReciprocalDiscriminator/out/Sigmoid,^gradients/add_grad/tuple/control_dependency*
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
4gradients/Discriminator/out/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out/Sigmoidgradients/Log_grad/mul*
T0*'
_output_shapes
:���������
[
gradients/sub_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
u
gradients/sub_grad/Shape_1ShapeDiscriminator/out_1/Sigmoid*
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
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*
_output_shapes
: *
T0
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
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
: *
T0
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*'
_output_shapes
:���������
�
4gradients/Discriminator/out/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
9gradients/Discriminator/out/BiasAdd_grad/tuple/group_depsNoOp5^gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad5^gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad
�
Agradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyIdentity4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������
�
Cgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid-gradients/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
.gradients/Discriminator/out/MatMul_grad/MatMulMatMulAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
0gradients/Discriminator/out/MatMul_grad/MatMul_1MatMul3Discriminator/second/Discriminator/secondleaky_reluAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
8gradients/Discriminator/out/MatMul_grad/tuple/group_depsNoOp/^gradients/Discriminator/out/MatMul_grad/MatMul1^gradients/Discriminator/out/MatMul_grad/MatMul_1
�
@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyIdentity.gradients/Discriminator/out/MatMul_grad/MatMul9^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Discriminator/out/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Bgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Identity0gradients/Discriminator/out/MatMul_grad/MatMul_19^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
_output_shapes
:	�*
T0
�
6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
;gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad7^gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
�
Cgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
�
Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad
�
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeShape7Discriminator/second/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Shape@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zerosFillJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Ogradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Xgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Igradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectSelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqual@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Kgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1SelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Fgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumSumIgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectXgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1SumKgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1Zgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Sgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeM^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1
�
[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeT^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1T^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
0gradients/Discriminator/out_1/MatMul_grad/MatMulMatMulCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
2gradients/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
�
:gradients/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp1^gradients/Discriminator/out_1/MatMul_grad/MatMul3^gradients/Discriminator/out_1/MatMul_grad/MatMul_1
�
Bgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity0gradients/Discriminator/out_1/MatMul_grad/MatMul;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*C
_class9
75loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul
�
Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity2gradients/Discriminator/out_1/MatMul_grad/MatMul_1;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
gradients/AddNAddNCgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/MulMul[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumSumJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul9Discriminator/second/Discriminator/secondleaky_relu/alpha[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Wgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1
�
_gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
�
agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
�
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Qgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Zgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Kgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Mgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Hgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumKgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectZgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeHgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumMgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1\gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ugradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpM^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeO^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
�
]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeV^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*_
_classU
SQloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape
�
_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1V^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_1AddNBgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
N*
_output_shapes
:	�
�
gradients/AddN_2AddN]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
data_formatNHWC*
_output_shapes	
:�*
T0
�
`gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_2\^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
hgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_2a^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1
�
jgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapePgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1`gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Rgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ygradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeS^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
�
agradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeZ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1Z^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Ugradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Wgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul1Discriminator/first/Discriminator/firstleaky_reluhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
_gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
�
ggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*h
_class^
\Zloc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul
�
igradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
�
gradients/AddN_3AddN_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������*
T0
�
]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
T0*
data_formatNHWC*
_output_shapes	
:�
�
bgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_3^^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
jgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3c^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradc^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*p
_classf
dbloc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeShape5Discriminator/first/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Shapeggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosFillHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Mgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Vgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ggradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectSelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Igradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1SelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Dgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumSumGgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectVgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeReshapeDgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1SumIgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1Xgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Jgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Qgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpI^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeK^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1
�
Ygradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeR^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1R^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Wgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMuljgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Ygradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relujgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
agradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulZ^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
�
igradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulb^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1b^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_4AddNjgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Zgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulMulYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumSumHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulZgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul7Discriminator/first/Discriminator/firstleaky_relu/alphaYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1\gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ngradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ugradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpM^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeO^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1
�
]gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeV^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*_
_classU
SQloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape
�
_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1V^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapeigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:����������*
T0
�
Ogradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Xgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Igradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Kgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Fgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumIgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectXgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumKgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1Zgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Sgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeM^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeT^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1T^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_5AddNigradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
��*
T0
�
gradients/AddN_6AddN[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ygradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*
data_formatNHWC*
_output_shapes	
:�
�
^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6Z^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
fgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
hgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Pgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Wgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
�
_gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*c
_classY
WUloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
Sgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Ugradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/realfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
]gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpT^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulV^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
egradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentitySgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
ggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityUgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_7AddN]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
data_formatNHWC*
_output_shapes	
:�*
T0
�
`gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_7\^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
hgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_7a^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Ugradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Wgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
�
_gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
ggradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*h
_class^
\Zloc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*j
_class`
^\loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
gradients/AddN_8AddNhgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes	
:�*
T0*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
gradients/AddN_9AddNggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
��
�
beta1_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_output_shapes
: *
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(
�
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
beta2_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(
�
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
dDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     
�
ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
TDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zerosFilldDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
BDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam
VariableV2* 
_output_shapes
:
��*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
��*
dtype0
�
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamTDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
GDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
�
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zerosFillfDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
DDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1
VariableV2* 
_output_shapes
:
��*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
��*
dtype0
�
KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignAssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/readIdentityDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
�
RDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zerosConst*
_output_shapes	
:�*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB�*    *
dtype0
�
@Discriminator/first/Discriminator/firstfully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:�
�
GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignAssign@Discriminator/first/Discriminator/firstfully_connected/bias/AdamRDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
EDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/readIdentity@Discriminator/first/Discriminator/firstfully_connected/bias/Adam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:�
�
TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
BDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container 
�
IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:�
�
fDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    
�
VDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zerosFillfDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensor\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0
�
DDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
�
KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamVDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
�
IDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��*
T0
�
hDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    
�
XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zerosFillhDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
FDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container 
�
MDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/AssignAssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
�
KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/readIdentityFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
�
TDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
BDiscriminator/second/Discriminator/secondfully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:�
�
IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/AssignAssignBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamTDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
GDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/readIdentityBDiscriminator/second/Discriminator/secondfully_connected/bias/Adam*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:�
�
VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
DDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:�
�
KDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zeros*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:�
�
/Discriminator/out/kernel/Adam/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
Discriminator/out/kernel/Adam
VariableV2*
_output_shapes
:	�*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	�*
dtype0
�
$Discriminator/out/kernel/Adam/AssignAssignDiscriminator/out/kernel/Adam/Discriminator/out/kernel/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	�
�
"Discriminator/out/kernel/Adam/readIdentityDiscriminator/out/kernel/Adam*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�
�
1Discriminator/out/kernel/Adam_1/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
Discriminator/out/kernel/Adam_1
VariableV2*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *+
_class!
loc:@Discriminator/out/kernel
�
&Discriminator/out/kernel/Adam_1/AssignAssignDiscriminator/out/kernel/Adam_11Discriminator/out/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel
�
$Discriminator/out/kernel/Adam_1/readIdentityDiscriminator/out/kernel/Adam_1*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�*
T0
�
-Discriminator/out/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*)
_class
loc:@Discriminator/out/bias*
valueB*    
�
Discriminator/out/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:
�
"Discriminator/out/bias/Adam/AssignAssignDiscriminator/out/bias/Adam-Discriminator/out/bias/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:
�
 Discriminator/out/bias/Adam/readIdentityDiscriminator/out/bias/Adam*
T0*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:
�
/Discriminator/out/bias/Adam_1/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
�
Discriminator/out/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias
�
$Discriminator/out/bias/Adam_1/AssignAssignDiscriminator/out/bias/Adam_1/Discriminator/out/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:
�
"Discriminator/out/bias/Adam_1/readIdentityDiscriminator/out/bias/Adam_1*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:*
T0
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

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
Q
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
SAdam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam	ApplyAdam=Discriminator/first/Discriminator/firstfully_connected/kernelBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
use_locking( *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
QAdam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdam	ApplyAdam;Discriminator/first/Discriminator/firstfully_connected/bias@Discriminator/first/Discriminator/firstfully_connected/bias/AdamBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
UAdam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam	ApplyAdam?Discriminator/second/Discriminator/secondfully_connected/kernelDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0
�
SAdam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdam	ApplyAdam=Discriminator/second/Discriminator/secondfully_connected/biasBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
.Adam/update_Discriminator/out/kernel/ApplyAdam	ApplyAdamDiscriminator/out/kernelDiscriminator/out/kernel/AdamDiscriminator/out/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*+
_class!
loc:@Discriminator/out/kernel*
use_nesterov( *
_output_shapes
:	�*
use_locking( *
T0
�
,Adam/update_Discriminator/out/bias/ApplyAdam	ApplyAdamDiscriminator/out/biasDiscriminator/out/bias/AdamDiscriminator/out/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
use_locking( *
T0*)
_class
loc:@Discriminator/out/bias*
use_nesterov( *
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_output_shapes
: *
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(
�
AdamNoOp^Adam/Assign^Adam/Assign_1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam
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
gradients_1/Mean_1_grad/ShapeShapeLog_2*
out_type0*
_output_shapes
:*
T0
�
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
d
gradients_1/Mean_1_grad/Shape_1ShapeLog_2*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_1_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
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
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
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
y
gradients_1/sub_1_grad/Shape_1ShapeDiscriminator/out_1/Sigmoid*
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
gradients_1/sub_1_grad/SumSumgradients_1/Log_2_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
_output_shapes
:*
T0
�
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
�
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*
_output_shapes
: *
T0
�
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1
�
8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid1gradients_1/sub_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
=gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp9^gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad9^gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
�
Egradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������
�
Ggradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
2gradients_1/Discriminator/out_1/MatMul_grad/MatMulMatMulEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
<gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp3^gradients_1/Discriminator/out_1/MatMul_grad/MatMul5^gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1
�
Dgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/Discriminator/out_1/MatMul_grad/MatMul=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*E
_class;
97loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul
�
Fgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:����������*
T0
�
Sgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Ogradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Jgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumMgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeJgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumOgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Wgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpO^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeQ^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
�
_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeX^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
agradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1X^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1bgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Tgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapePgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
[gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpS^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeU^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
�
cgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*e
_class[
YWloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape
�
egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddNAddNagradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes	
:�
�
dgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN`^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
lgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddNe^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
ngradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrade^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*r
_classh
fdloc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
Ygradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMullgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relulgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
cgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpZ^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul\^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
�
kgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityYgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMuld^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
mgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1Identity[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1d^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*n
_classd
b`loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
�
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapekgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
�
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Qgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Zgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Kgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualkgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Mgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeroskgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Hgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumKgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectZgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeHgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumMgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1\gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ugradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpM^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeO^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeV^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1V^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapePgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1`gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Rgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ygradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeS^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
�
agradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeZ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1Z^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_1AddN_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:�
�
bgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1^^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
jgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1c^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
lgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradc^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Wgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMuljgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Ygradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhjgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
agradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulZ^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
igradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulb^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*j
_class`
^\loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul
�
kgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1b^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*l
_classb
`^loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
(gradients_1/Generator/Tanh_grad/TanhGradTanhGradGenerator/Tanhigradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/Generator/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:�*
T0
�
9gradients_1/Generator/dense/BiasAdd_grad/tuple/group_depsNoOp)^gradients_1/Generator/Tanh_grad/TanhGrad5^gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad
�
Agradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/Generator/Tanh_grad/TanhGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/Generator/Tanh_grad/TanhGrad*(
_output_shapes
:����������
�
Cgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
.gradients_1/Generator/dense/MatMul_grad/MatMulMatMulAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyGenerator/dense/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
0gradients_1/Generator/dense/MatMul_grad/MatMul_1MatMul)Generator/forth/Generator/forthleaky_reluAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
8gradients_1/Generator/dense/MatMul_grad/tuple/group_depsNoOp/^gradients_1/Generator/dense/MatMul_grad/MatMul1^gradients_1/Generator/dense/MatMul_grad/MatMul_1
�
@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/Generator/dense/MatMul_grad/MatMul9^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/Generator/dense/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Bgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/Generator/dense/MatMul_grad/MatMul_19^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*C
_class9
75loc:@gradients_1/Generator/dense/MatMul_grad/MatMul_1
�
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeShape-Generator/forth/Generator/forthleaky_relu/mul*
out_type0*
_output_shapes
:*
T0
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Shape@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zerosFillBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Ggradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqualGreaterEqual-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Pgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Agradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectSelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Cgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1SelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/SumSumAgradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectPgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeReshape>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1SumCgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1Rgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Kgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeE^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeL^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*U
_classK
IGloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape
�
Ugradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1L^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
Tgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulMulSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumSumBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulTgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Mul/Generator/forth/Generator/forthleaky_relu/alphaSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Hgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1
�
Wgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
gradients_1/AddN_2AddNUgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������*
T0
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_2fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_2hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/forth/Generator/forthfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAddigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg
�
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Sgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Xgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad
�
`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape
�
bgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:�
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/forth/Generator/forthbatch_normalized/moving_mean/read*
_output_shapes	
:�*
T0
�
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul
�
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Mgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/forth/Generator/forthfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Ogradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1MatMul)Generator/third/Generator/thirdleaky_relu`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Wgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1
�
_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
agradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_3AddNkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N
�
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_3:Generator/forth/Generator/forthbatch_normalized/gamma/read*
T0*
_output_shapes	
:�
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_3?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt*
_output_shapes	
:�*
T0
�
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1
�
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeShape-Generator/third/Generator/thirdleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Shape_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zerosFillBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Ggradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqualGreaterEqual-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Pgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Agradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectSelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Cgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1SelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/SumSumAgradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectPgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeReshape>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1SumCgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1Rgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Kgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeE^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeL^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Ugradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1L^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Tgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulMulSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumSumBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulTgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Mul/Generator/third/Generator/thirdleaky_relu/alphaSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Hgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Ogradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1
�
Wgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_4AddNUgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_4fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/third/Generator/thirdfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAddigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg
�
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Sgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Xgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad
�
`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape
�
bgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
_output_shapes	
:�*
T0
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/third/Generator/thirdbatch_normalized/moving_mean/read*
_output_shapes	
:�*
T0
�
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
�
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*i
_class_
][loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1
�
Mgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/third/Generator/thirdfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Ogradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1MatMul+Generator/second/Generator/secondleaky_relu`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Wgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1
�
_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
agradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*b
_classX
VTloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1
�
gradients_1/AddN_5AddNkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N
�
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_5:Generator/third/Generator/thirdbatch_normalized/gamma/read*
_output_shapes	
:�*
T0
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_5?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt*
_output_shapes	
:�*
T0
�
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1
�
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeShape/Generator/second/Generator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Shape_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/zerosFillDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Igradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualGreaterEqual/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Rgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Cgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectSelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqual_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Egradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1SelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumSumCgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectRgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeReshape@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1SumEgradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1Tgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Fgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1ReshapeBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Mgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_depsNoOpE^gradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeG^gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1
�
Ugradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyIdentityDgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeN^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape
�
Wgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1N^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Vgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulMulUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumSumDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulVgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeReshapeDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Mul1Generator/second/Generator/secondleaky_relu/alphaUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Xgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Jgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1ReshapeFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Qgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_depsNoOpI^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeK^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1
�
Ygradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeR^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1R^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_6AddNWgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_6hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape
�
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeShape8Generator/second/Generator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumSumVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mulhgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAddkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/NegNegmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpn^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1U^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg
�
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitymgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Negb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Ugradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Zgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOpl^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyV^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
bgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitykgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*h
_class^
\Zloc:@gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:�
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1Mulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1BGenerator/second/Generator/secondbatch_normalized/moving_mean/read*
_output_shapes	
:�*
T0
�
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulY^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Muld^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
�
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Ogradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulMatMulbgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency<Generator/second/Generator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Qgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1MatMul)Generator/first/Generator/firstleaky_relubgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
Ygradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpP^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulR^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1
�
agradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulZ^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*b
_classX
VTloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul
�
cgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1Z^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*d
_classZ
XVloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1
�
gradients_1/AddN_7AddNmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_7<Generator/second/Generator/secondbatch_normalized/gamma/read*
_output_shapes	
:�*
T0
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_7AGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpU^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1
�
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mulb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�*
T0
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeShape-Generator/first/Generator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Shapeagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosFillBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Ggradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualGreaterEqual-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Pgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Agradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectSelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Cgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1SelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqual@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
>gradients_1/Generator/first/Generator/firstleaky_relu_grad/SumSumAgradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectPgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeReshape>gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1SumCgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1Rgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Dgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Kgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeE^gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeL^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Ugradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1L^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Tgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulMulSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency6Generator/first/Generator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumSumBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulTgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Mul/Generator/first/Generator/firstleaky_relu/alphaSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Hgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Ogradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1
�
Wgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*[
_classQ
OMloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1
�
gradients_1/AddN_8AddNUgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Xgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_8T^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8Y^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
bgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Mgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/first/Generator/firstfully_connected/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
�
Ogradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d�*
transpose_a(*
transpose_b( 
�
Wgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1
�
_gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*`
_classV
TRloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul
�
agradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
_output_shapes
:	d�*
T0*b
_classX
VTloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1
�
beta1_power_1/initial_valueConst*
_output_shapes
: *'
_class
loc:@Generator/dense/bias*
valueB
 *fff?*
dtype0
�
beta1_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape: 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
w
beta1_power_1/readIdentitybeta1_power_1*'
_class
loc:@Generator/dense/bias*
_output_shapes
: *
T0
�
beta2_power_1/initial_valueConst*
dtype0*
_output_shapes
: *'
_class
loc:@Generator/dense/bias*
valueB
 *w�?
�
beta2_power_1
VariableV2*
shared_name *'
_class
loc:@Generator/dense/bias*
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
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(
w
beta2_power_1/readIdentitybeta2_power_1*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
�
\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
�
RGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zerosFill\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
:Generator/first/Generator/firstfully_connected/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	d�*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d�
�
AGenerator/first/Generator/firstfully_connected/kernel/Adam/AssignAssign:Generator/first/Generator/firstfully_connected/kernel/AdamLGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
?Generator/first/Generator/firstfully_connected/kernel/Adam/readIdentity:Generator/first/Generator/firstfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�
�
^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   �   
�
TGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
<Generator/first/Generator/firstfully_connected/kernel/Adam_1
VariableV2*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
�
CGenerator/first/Generator/firstfully_connected/kernel/Adam_1/AssignAssign<Generator/first/Generator/firstfully_connected/kernel/Adam_1NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0
�
AGenerator/first/Generator/firstfully_connected/kernel/Adam_1/readIdentity<Generator/first/Generator/firstfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�
�
JGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
8Generator/first/Generator/firstfully_connected/bias/Adam
VariableV2*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
?Generator/first/Generator/firstfully_connected/bias/Adam/AssignAssign8Generator/first/Generator/firstfully_connected/bias/AdamJGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
=Generator/first/Generator/firstfully_connected/bias/Adam/readIdentity8Generator/first/Generator/firstfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:�
�
LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
:Generator/first/Generator/firstfully_connected/bias/Adam_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
�
AGenerator/first/Generator/firstfully_connected/bias/Adam_1/AssignAssign:Generator/first/Generator/firstfully_connected/bias/Adam_1LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
?Generator/first/Generator/firstfully_connected/bias/Adam_1/readIdentity:Generator/first/Generator/firstfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:�
�
^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
TGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    
�
NGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zerosFill^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorTGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
<Generator/second/Generator/secondfully_connected/kernel/Adam
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container 
�
CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignAssign<Generator/second/Generator/secondfully_connected/kernel/AdamNGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
AGenerator/second/Generator/secondfully_connected/kernel/Adam/readIdentity<Generator/second/Generator/secondfully_connected/kernel/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
�
`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0
�
VGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zerosFill`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorVGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
>Generator/second/Generator/secondfully_connected/kernel/Adam_1
VariableV2*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignAssign>Generator/second/Generator/secondfully_connected/kernel/Adam_1PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
CGenerator/second/Generator/secondfully_connected/kernel/Adam_1/readIdentity>Generator/second/Generator/secondfully_connected/kernel/Adam_1*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��*
T0
�
LGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
:Generator/second/Generator/secondfully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:�
�
AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAssign:Generator/second/Generator/secondfully_connected/bias/AdamLGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
?Generator/second/Generator/secondfully_connected/bias/Adam/readIdentity:Generator/second/Generator/secondfully_connected/bias/Adam*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:�*
T0
�
NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
<Generator/second/Generator/secondfully_connected/bias/Adam_1
VariableV2*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignAssign<Generator/second/Generator/secondfully_connected/bias/Adam_1NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
AGenerator/second/Generator/secondfully_connected/bias/Adam_1/readIdentity<Generator/second/Generator/secondfully_connected/bias/Adam_1*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
�
NGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
<Generator/second/Generator/secondbatch_normalized/gamma/Adam
VariableV2*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam/AssignAssign<Generator/second/Generator/secondbatch_normalized/gamma/AdamNGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zeros*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
AGenerator/second/Generator/secondbatch_normalized/gamma/Adam/readIdentity<Generator/second/Generator/secondbatch_normalized/gamma/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:�
�
PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB�*    *
dtype0
�
>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1
VariableV2*
_output_shapes	
:�*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:�*
dtype0
�
EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignAssign>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/readIdentity>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1*
_output_shapes	
:�*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma
�
MGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
;Generator/second/Generator/secondbatch_normalized/beta/Adam
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
�
BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignAssign;Generator/second/Generator/secondbatch_normalized/beta/AdamMGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
�
@Generator/second/Generator/secondbatch_normalized/beta/Adam/readIdentity;Generator/second/Generator/secondbatch_normalized/beta/Adam*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:�
�
OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
=Generator/second/Generator/secondbatch_normalized/beta/Adam_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container 
�
DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignAssign=Generator/second/Generator/secondbatch_normalized/beta/Adam_1OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
BGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/readIdentity=Generator/second/Generator/secondbatch_normalized/beta/Adam_1*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:�
�
\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
RGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zerosFill\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
:Generator/third/Generator/thirdfully_connected/kernel/Adam
VariableV2*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
AGenerator/third/Generator/thirdfully_connected/kernel/Adam/AssignAssign:Generator/third/Generator/thirdfully_connected/kernel/AdamLGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
?Generator/third/Generator/thirdfully_connected/kernel/Adam/readIdentity:Generator/third/Generator/thirdfully_connected/kernel/Adam*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��*
T0
�
^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
TGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
<Generator/third/Generator/thirdfully_connected/kernel/Adam_1
VariableV2*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignAssign<Generator/third/Generator/thirdfully_connected/kernel/Adam_1NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(
�
AGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/readIdentity<Generator/third/Generator/thirdfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��
�
JGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
8Generator/third/Generator/thirdfully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:�
�
?Generator/third/Generator/thirdfully_connected/bias/Adam/AssignAssign8Generator/third/Generator/thirdfully_connected/bias/AdamJGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
�
=Generator/third/Generator/thirdfully_connected/bias/Adam/readIdentity8Generator/third/Generator/thirdfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:�
�
LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB�*    *
dtype0
�
:Generator/third/Generator/thirdfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:�
�
AGenerator/third/Generator/thirdfully_connected/bias/Adam_1/AssignAssign:Generator/third/Generator/thirdfully_connected/bias/Adam_1LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
?Generator/third/Generator/thirdfully_connected/bias/Adam_1/readIdentity:Generator/third/Generator/thirdfully_connected/bias/Adam_1*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:�*
T0
�
LGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zerosConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
:Generator/third/Generator/thirdbatch_normalized/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:�
�
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAssign:Generator/third/Generator/thirdbatch_normalized/gamma/AdamLGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
?Generator/third/Generator/thirdbatch_normalized/gamma/Adam/readIdentity:Generator/third/Generator/thirdbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:�
�
NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:�
�
CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignAssign<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/readIdentity<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:�
�
KGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
9Generator/third/Generator/thirdbatch_normalized/beta/Adam
VariableV2*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
@Generator/third/Generator/thirdbatch_normalized/beta/Adam/AssignAssign9Generator/third/Generator/thirdbatch_normalized/beta/AdamKGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
�
>Generator/third/Generator/thirdbatch_normalized/beta/Adam/readIdentity9Generator/third/Generator/thirdbatch_normalized/beta/Adam*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:�
�
MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB�*    
�
;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
�
BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignAssign;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(
�
@Generator/third/Generator/thirdbatch_normalized/beta/Adam_1/readIdentity;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:�
�
\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
RGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0
�
:Generator/forth/Generator/forthfully_connected/kernel/Adam
VariableV2*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAssign:Generator/forth/Generator/forthfully_connected/kernel/AdamLGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
?Generator/forth/Generator/forthfully_connected/kernel/Adam/readIdentity:Generator/forth/Generator/forthfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��
�
^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0
�
TGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/Const*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
<Generator/forth/Generator/forthfully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container *
shape:
��
�
CGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/AssignAssign<Generator/forth/Generator/forthfully_connected/kernel/Adam_1NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
AGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/readIdentity<Generator/forth/Generator/forthfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��
�
ZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
PGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
JGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zerosFillZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorPGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/Const*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0
�
8Generator/forth/Generator/forthfully_connected/bias/Adam
VariableV2*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
?Generator/forth/Generator/forthfully_connected/bias/Adam/AssignAssign8Generator/forth/Generator/forthfully_connected/bias/AdamJGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
=Generator/forth/Generator/forthfully_connected/bias/Adam/readIdentity8Generator/forth/Generator/forthfully_connected/bias/Adam*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
�
\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�*
dtype0
�
RGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:�
�
:Generator/forth/Generator/forthfully_connected/bias/Adam_1
VariableV2*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAssign:Generator/forth/Generator/forthfully_connected/bias/Adam_1LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
?Generator/forth/Generator/forthfully_connected/bias/Adam_1/readIdentity:Generator/forth/Generator/forthfully_connected/bias/Adam_1*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:�*
T0
�
\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:�
�
RGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    
�
LGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zerosFill\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:�
�
:Generator/forth/Generator/forthbatch_normalized/gamma/Adam
VariableV2*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAssign:Generator/forth/Generator/forthbatch_normalized/gamma/AdamLGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
?Generator/forth/Generator/forthbatch_normalized/gamma/Adam/readIdentity:Generator/forth/Generator/forthbatch_normalized/gamma/Adam*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
�
^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:�
�
TGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:�
�
<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1
VariableV2*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignAssign<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/readIdentity<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
�
[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:�*
dtype0*
_output_shapes
:
�
QGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    
�
KGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zerosFill[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorQGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/Const*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:�*
T0
�
9Generator/forth/Generator/forthbatch_normalized/beta/Adam
VariableV2*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
@Generator/forth/Generator/forthbatch_normalized/beta/Adam/AssignAssign9Generator/forth/Generator/forthbatch_normalized/beta/AdamKGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
�
>Generator/forth/Generator/forthbatch_normalized/beta/Adam/readIdentity9Generator/forth/Generator/forthbatch_normalized/beta/Adam*
_output_shapes	
:�*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
�
]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:�*
dtype0*
_output_shapes
:
�
SGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/Const*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:�*
T0
�
;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1
VariableV2*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignAssign;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
@Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/readIdentity;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:�*
T0
�
=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
3Generator/dense/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
-Generator/dense/kernel/Adam/Initializer/zerosFill=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3Generator/dense/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*)
_class
loc:@Generator/dense/kernel*

index_type0
�
Generator/dense/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:
��
�
"Generator/dense/kernel/Adam/AssignAssignGenerator/dense/kernel/Adam-Generator/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
��
�
 Generator/dense/kernel/Adam/readIdentityGenerator/dense/kernel/Adam* 
_output_shapes
:
��*
T0*)
_class
loc:@Generator/dense/kernel
�
?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
5Generator/dense/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/Generator/dense/kernel/Adam_1/Initializer/zerosFill?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5Generator/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@Generator/dense/kernel*

index_type0* 
_output_shapes
:
��
�
Generator/dense/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:
��
�
$Generator/dense/kernel/Adam_1/AssignAssignGenerator/dense/kernel/Adam_1/Generator/dense/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(
�
"Generator/dense/kernel/Adam_1/readIdentityGenerator/dense/kernel/Adam_1*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
��
�
+Generator/dense/bias/Adam/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Generator/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:�
�
 Generator/dense/bias/Adam/AssignAssignGenerator/dense/bias/Adam+Generator/dense/bias/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(
�
Generator/dense/bias/Adam/readIdentityGenerator/dense/bias/Adam*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:�*
T0
�
-Generator/dense/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Generator/dense/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
"Generator/dense/bias/Adam_1/AssignAssignGenerator/dense/bias/Adam_1-Generator/dense/bias/Adam_1/Initializer/zeros*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
 Generator/dense/bias/Adam_1/readIdentityGenerator/dense/bias/Adam_1*
_output_shapes	
:�*
T0*'
_class
loc:@Generator/dense/bias
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
MAdam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/first/Generator/firstfully_connected/kernel:Generator/first/Generator/firstfully_connected/kernel/Adam<Generator/first/Generator/firstfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
use_nesterov( *
_output_shapes
:	d�
�
KAdam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdam	ApplyAdam3Generator/first/Generator/firstfully_connected/bias8Generator/first/Generator/firstfully_connected/bias/Adam:Generator/first/Generator/firstfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
OAdam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdam	ApplyAdam7Generator/second/Generator/secondfully_connected/kernel<Generator/second/Generator/secondfully_connected/kernel/Adam>Generator/second/Generator/secondfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
MAdam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdam	ApplyAdam5Generator/second/Generator/secondfully_connected/bias:Generator/second/Generator/secondfully_connected/bias/Adam<Generator/second/Generator/secondfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:�*
use_locking( *
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
use_nesterov( 
�
OAdam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdam	ApplyAdam7Generator/second/Generator/secondbatch_normalized/gamma<Generator/second/Generator/secondbatch_normalized/gamma/Adam>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
NAdam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdam	ApplyAdam6Generator/second/Generator/secondbatch_normalized/beta;Generator/second/Generator/secondbatch_normalized/beta/Adam=Generator/second/Generator/secondbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:�
�
MAdam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdfully_connected/kernel:Generator/third/Generator/thirdfully_connected/kernel/Adam<Generator/third/Generator/thirdfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
KAdam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdam	ApplyAdam3Generator/third/Generator/thirdfully_connected/bias8Generator/third/Generator/thirdfully_connected/bias/Adam:Generator/third/Generator/thirdfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
MAdam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdbatch_normalized/gamma:Generator/third/Generator/thirdbatch_normalized/gamma/Adam<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
LAdam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/third/Generator/thirdbatch_normalized/beta9Generator/third/Generator/thirdbatch_normalized/beta/Adam;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
�
MAdam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthfully_connected/kernel:Generator/forth/Generator/forthfully_connected/kernel/Adam<Generator/forth/Generator/forthfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
KAdam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdam	ApplyAdam3Generator/forth/Generator/forthfully_connected/bias8Generator/forth/Generator/forthfully_connected/bias/Adam:Generator/forth/Generator/forthfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
MAdam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthbatch_normalized/gamma:Generator/forth/Generator/forthbatch_normalized/gamma/Adam<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:�
�
LAdam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/forth/Generator/forthbatch_normalized/beta9Generator/forth/Generator/forthbatch_normalized/beta/Adam;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes	
:�*
use_locking( *
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
use_nesterov( 
�
.Adam_1/update_Generator/dense/kernel/ApplyAdam	ApplyAdamGenerator/dense/kernelGenerator/dense/kernel/AdamGenerator/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1*
T0*)
_class
loc:@Generator/dense/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
�
,Adam_1/update_Generator/dense/bias/ApplyAdam	ApplyAdamGenerator/dense/biasGenerator/dense/bias/AdamGenerator/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1*'
_class
loc:@Generator/dense/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*'
_class
loc:@Generator/dense/bias*
_output_shapes
: *
T0
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*'
_class
loc:@Generator/dense/bias
�

Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
�	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: "_S~o�b     ��x6	���^���AJ��
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
shared_namestring �*1.12.02v1.12.0-0-ga6d8ffae09��
r
Generator/noisePlaceholder*
dtype0*'
_output_shapes
:���������d*
shape:���������d
�
VGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
�
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&�*
dtype0*
_output_shapes
: 
�
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&>*
dtype0
�
^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d�*

seed*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
seed2
�
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
�
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�
�
PGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniformAddTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�
�
5Generator/first/Generator/firstfully_connected/kernel
VariableV2*
_output_shapes
:	d�*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d�*
dtype0
�
<Generator/first/Generator/firstfully_connected/kernel/AssignAssign5Generator/first/Generator/firstfully_connected/kernelPGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
:Generator/first/Generator/firstfully_connected/kernel/readIdentity5Generator/first/Generator/firstfully_connected/kernel*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�*
T0
�
EGenerator/first/Generator/firstfully_connected/bias/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3Generator/first/Generator/firstfully_connected/bias
VariableV2*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
:Generator/first/Generator/firstfully_connected/bias/AssignAssign3Generator/first/Generator/firstfully_connected/biasEGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
8Generator/first/Generator/firstfully_connected/bias/readIdentity3Generator/first/Generator/firstfully_connected/bias*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:�
�
5Generator/first/Generator/firstfully_connected/MatMulMatMulGenerator/noise:Generator/first/Generator/firstfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
6Generator/first/Generator/firstfully_connected/BiasAddBiasAdd5Generator/first/Generator/firstfully_connected/MatMul8Generator/first/Generator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
t
/Generator/first/Generator/firstleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
-Generator/first/Generator/firstleaky_relu/mulMul/Generator/first/Generator/firstleaky_relu/alpha6Generator/first/Generator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
)Generator/first/Generator/firstleaky_reluMaximum-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
XGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/minConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   �*
dtype0*
_output_shapes
: 
�
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   >
�
`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformXGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
seed2
�
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/subSubVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
_output_shapes
: 
�
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulMul`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
�
RGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniformAddVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
�
7Generator/second/Generator/secondfully_connected/kernel
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container 
�
>Generator/second/Generator/secondfully_connected/kernel/AssignAssign7Generator/second/Generator/secondfully_connected/kernelRGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
�
<Generator/second/Generator/secondfully_connected/kernel/readIdentity7Generator/second/Generator/secondfully_connected/kernel*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
�
GGenerator/second/Generator/secondfully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB�*    
�
5Generator/second/Generator/secondfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:�
�
<Generator/second/Generator/secondfully_connected/bias/AssignAssign5Generator/second/Generator/secondfully_connected/biasGGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
�
:Generator/second/Generator/secondfully_connected/bias/readIdentity5Generator/second/Generator/secondfully_connected/bias*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:�*
T0
�
7Generator/second/Generator/secondfully_connected/MatMulMatMul)Generator/first/Generator/firstleaky_relu<Generator/second/Generator/secondfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
8Generator/second/Generator/secondfully_connected/BiasAddBiasAdd7Generator/second/Generator/secondfully_connected/MatMul:Generator/second/Generator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
HGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/onesConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
7Generator/second/Generator/secondbatch_normalized/gamma
VariableV2*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
>Generator/second/Generator/secondbatch_normalized/gamma/AssignAssign7Generator/second/Generator/secondbatch_normalized/gammaHGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
<Generator/second/Generator/secondbatch_normalized/gamma/readIdentity7Generator/second/Generator/secondbatch_normalized/gamma*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:�*
T0
�
HGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
6Generator/second/Generator/secondbatch_normalized/beta
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
�
=Generator/second/Generator/secondbatch_normalized/beta/AssignAssign6Generator/second/Generator/secondbatch_normalized/betaHGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
;Generator/second/Generator/secondbatch_normalized/beta/readIdentity6Generator/second/Generator/secondbatch_normalized/beta*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:�
�
OGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zerosConst*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
=Generator/second/Generator/secondbatch_normalized/moving_mean
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean
�
DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignAssign=Generator/second/Generator/secondbatch_normalized/moving_meanOGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
BGenerator/second/Generator/secondbatch_normalized/moving_mean/readIdentity=Generator/second/Generator/secondbatch_normalized/moving_mean*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
_output_shapes	
:�
�
RGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/onesConst*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
AGenerator/second/Generator/secondbatch_normalized/moving_variance
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
	container *
shape:�
�
HGenerator/second/Generator/secondbatch_normalized/moving_variance/AssignAssignAGenerator/second/Generator/secondbatch_normalized/moving_varianceRGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/ones*
_output_shapes	
:�*
use_locking(*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
validate_shape(
�
FGenerator/second/Generator/secondbatch_normalized/moving_variance/readIdentityAGenerator/second/Generator/secondbatch_normalized/moving_variance*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
_output_shapes	
:�
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
?Generator/second/Generator/secondbatch_normalized/batchnorm/addAddFGenerator/second/Generator/secondbatch_normalized/moving_variance/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:�
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/RsqrtRsqrt?Generator/second/Generator/secondbatch_normalized/batchnorm/add*
_output_shapes	
:�*
T0
�
?Generator/second/Generator/secondbatch_normalized/batchnorm/mulMulAGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt<Generator/second/Generator/secondbatch_normalized/gamma/read*
T0*
_output_shapes	
:�
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAdd?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*(
_output_shapes
:����������*
T0
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2MulBGenerator/second/Generator/secondbatch_normalized/moving_mean/read?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:�
�
?Generator/second/Generator/secondbatch_normalized/batchnorm/subSub;Generator/second/Generator/secondbatch_normalized/beta/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1AddAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1?Generator/second/Generator/secondbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:����������
v
1Generator/second/Generator/secondleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
/Generator/second/Generator/secondleaky_relu/mulMul1Generator/second/Generator/secondleaky_relu/alphaAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
+Generator/second/Generator/secondleaky_reluMaximum/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
VGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *��=*
dtype0*
_output_shapes
: 
�
^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
seed2B
�
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
�
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
�
PGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniformAddTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��
�
5Generator/third/Generator/thirdfully_connected/kernel
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
�
<Generator/third/Generator/thirdfully_connected/kernel/AssignAssign5Generator/third/Generator/thirdfully_connected/kernelPGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
��*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(
�
:Generator/third/Generator/thirdfully_connected/kernel/readIdentity5Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
�
EGenerator/third/Generator/thirdfully_connected/bias/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3Generator/third/Generator/thirdfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:�
�
:Generator/third/Generator/thirdfully_connected/bias/AssignAssign3Generator/third/Generator/thirdfully_connected/biasEGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
8Generator/third/Generator/thirdfully_connected/bias/readIdentity3Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
�
5Generator/third/Generator/thirdfully_connected/MatMulMatMul+Generator/second/Generator/secondleaky_relu:Generator/third/Generator/thirdfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
6Generator/third/Generator/thirdfully_connected/BiasAddBiasAdd5Generator/third/Generator/thirdfully_connected/MatMul8Generator/third/Generator/thirdfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
�
FGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:�*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB�*  �?
�
5Generator/third/Generator/thirdbatch_normalized/gamma
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:�
�
<Generator/third/Generator/thirdbatch_normalized/gamma/AssignAssign5Generator/third/Generator/thirdbatch_normalized/gammaFGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(
�
:Generator/third/Generator/thirdbatch_normalized/gamma/readIdentity5Generator/third/Generator/thirdbatch_normalized/gamma*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:�
�
FGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zerosConst*
_output_shapes	
:�*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB�*    *
dtype0
�
4Generator/third/Generator/thirdbatch_normalized/beta
VariableV2*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:�*
dtype0
�
;Generator/third/Generator/thirdbatch_normalized/beta/AssignAssign4Generator/third/Generator/thirdbatch_normalized/betaFGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
9Generator/third/Generator/thirdbatch_normalized/beta/readIdentity4Generator/third/Generator/thirdbatch_normalized/beta*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:�
�
MGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zerosConst*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
;Generator/third/Generator/thirdbatch_normalized/moving_mean
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
	container 
�
BGenerator/third/Generator/thirdbatch_normalized/moving_mean/AssignAssign;Generator/third/Generator/thirdbatch_normalized/moving_meanMGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zeros*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
@Generator/third/Generator/thirdbatch_normalized/moving_mean/readIdentity;Generator/third/Generator/thirdbatch_normalized/moving_mean*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
_output_shapes	
:�
�
PGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/onesConst*
_output_shapes	
:�*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
valueB�*  �?*
dtype0
�
?Generator/third/Generator/thirdbatch_normalized/moving_variance
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
	container *
shape:�
�
FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignAssign?Generator/third/Generator/thirdbatch_normalized/moving_variancePGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
DGenerator/third/Generator/thirdbatch_normalized/moving_variance/readIdentity?Generator/third/Generator/thirdbatch_normalized/moving_variance*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
_output_shapes	
:�*
T0
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
=Generator/third/Generator/thirdbatch_normalized/batchnorm/addAddDGenerator/third/Generator/thirdbatch_normalized/moving_variance/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:�
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/RsqrtRsqrt=Generator/third/Generator/thirdbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:�
�
=Generator/third/Generator/thirdbatch_normalized/batchnorm/mulMul?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:�
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAdd=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*(
_output_shapes
:����������*
T0
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2Mul@Generator/third/Generator/thirdbatch_normalized/moving_mean/read=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:�
�
=Generator/third/Generator/thirdbatch_normalized/batchnorm/subSub9Generator/third/Generator/thirdbatch_normalized/beta/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1Add?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:����������
t
/Generator/third/Generator/thirdleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *��L>*
dtype0
�
-Generator/third/Generator/thirdleaky_relu/mulMul/Generator/third/Generator/thirdleaky_relu/alpha?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
)Generator/third/Generator/thirdleaky_reluMaximum-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
VGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  ��*
dtype0*
_output_shapes
: 
�
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  �=
�
^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shape*
seed2m*
dtype0* 
_output_shapes
:
��*

seed*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
�
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
�
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��
�
PGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniformAddTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��
�
5Generator/forth/Generator/forthfully_connected/kernel
VariableV2*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
<Generator/forth/Generator/forthfully_connected/kernel/AssignAssign5Generator/forth/Generator/forthfully_connected/kernelPGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
:Generator/forth/Generator/forthfully_connected/kernel/readIdentity5Generator/forth/Generator/forthfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��
�
UGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
KGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
EGenerator/forth/Generator/forthfully_connected/bias/Initializer/zerosFillUGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorKGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:�
�
3Generator/forth/Generator/forthfully_connected/bias
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container 
�
:Generator/forth/Generator/forthfully_connected/bias/AssignAssign3Generator/forth/Generator/forthfully_connected/biasEGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
8Generator/forth/Generator/forthfully_connected/bias/readIdentity3Generator/forth/Generator/forthfully_connected/bias*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:�
�
5Generator/forth/Generator/forthfully_connected/MatMulMatMul)Generator/third/Generator/thirdleaky_relu:Generator/forth/Generator/forthfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
6Generator/forth/Generator/forthfully_connected/BiasAddBiasAdd5Generator/forth/Generator/forthfully_connected/MatMul8Generator/forth/Generator/forthfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
VGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:�*
dtype0*
_output_shapes
:
�
LGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
FGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/onesFillVGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/Const*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0
�
5Generator/forth/Generator/forthbatch_normalized/gamma
VariableV2*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
<Generator/forth/Generator/forthbatch_normalized/gamma/AssignAssign5Generator/forth/Generator/forthbatch_normalized/gammaFGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
�
:Generator/forth/Generator/forthbatch_normalized/gamma/readIdentity5Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
�
VGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:�*
dtype0*
_output_shapes
:
�
LGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
FGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zerosFillVGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:�
�
4Generator/forth/Generator/forthbatch_normalized/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:�
�
;Generator/forth/Generator/forthbatch_normalized/beta/AssignAssign4Generator/forth/Generator/forthbatch_normalized/betaFGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
�
9Generator/forth/Generator/forthbatch_normalized/beta/readIdentity4Generator/forth/Generator/forthbatch_normalized/beta*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:�
�
]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorConst*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB:�*
dtype0*
_output_shapes
:
�
SGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/ConstConst*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
MGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/Const*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*

index_type0*
_output_shapes	
:�
�
;Generator/forth/Generator/forthbatch_normalized/moving_mean
VariableV2*
shared_name *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignAssign;Generator/forth/Generator/forthbatch_normalized/moving_meanMGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:�
�
@Generator/forth/Generator/forthbatch_normalized/moving_mean/readIdentity;Generator/forth/Generator/forthbatch_normalized/moving_mean*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
_output_shapes	
:�
�
`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorConst*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB:�*
dtype0*
_output_shapes
:
�
VGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB
 *  �?
�
PGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/onesFill`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorVGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/Const*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*

index_type0*
_output_shapes	
:�
�
?Generator/forth/Generator/forthbatch_normalized/moving_variance
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
	container 
�
FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignAssign?Generator/forth/Generator/forthbatch_normalized/moving_variancePGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones*
use_locking(*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:�
�
DGenerator/forth/Generator/forthbatch_normalized/moving_variance/readIdentity?Generator/forth/Generator/forthbatch_normalized/moving_variance*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
_output_shapes	
:�
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
=Generator/forth/Generator/forthbatch_normalized/batchnorm/addAddDGenerator/forth/Generator/forthbatch_normalized/moving_variance/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/y*
_output_shapes	
:�*
T0
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/RsqrtRsqrt=Generator/forth/Generator/forthbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:�
�
=Generator/forth/Generator/forthbatch_normalized/batchnorm/mulMul?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt:Generator/forth/Generator/forthbatch_normalized/gamma/read*
_output_shapes	
:�*
T0
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAdd=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2Mul@Generator/forth/Generator/forthbatch_normalized/moving_mean/read=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:�
�
=Generator/forth/Generator/forthbatch_normalized/batchnorm/subSub9Generator/forth/Generator/forthbatch_normalized/beta/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1Add?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:����������
t
/Generator/forth/Generator/forthleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
-Generator/forth/Generator/forthleaky_relu/mulMul/Generator/forth/Generator/forthleaky_relu/alpha?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
)Generator/forth/Generator/forthleaky_reluMaximum-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
7Generator/dense/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
5Generator/dense/kernel/Initializer/random_uniform/minConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *z�k�*
dtype0*
_output_shapes
: 
�
5Generator/dense/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *z�k=*
dtype0*
_output_shapes
: 
�
?Generator/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7Generator/dense/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed*
T0*)
_class
loc:@Generator/dense/kernel*
seed2�
�
5Generator/dense/kernel/Initializer/random_uniform/subSub5Generator/dense/kernel/Initializer/random_uniform/max5Generator/dense/kernel/Initializer/random_uniform/min*)
_class
loc:@Generator/dense/kernel*
_output_shapes
: *
T0
�
5Generator/dense/kernel/Initializer/random_uniform/mulMul?Generator/dense/kernel/Initializer/random_uniform/RandomUniform5Generator/dense/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*)
_class
loc:@Generator/dense/kernel
�
1Generator/dense/kernel/Initializer/random_uniformAdd5Generator/dense/kernel/Initializer/random_uniform/mul5Generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
��
�
Generator/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:
��
�
Generator/dense/kernel/AssignAssignGenerator/dense/kernel1Generator/dense/kernel/Initializer/random_uniform* 
_output_shapes
:
��*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(
�
Generator/dense/kernel/readIdentityGenerator/dense/kernel*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
��
�
&Generator/dense/bias/Initializer/zerosConst*
_output_shapes	
:�*'
_class
loc:@Generator/dense/bias*
valueB�*    *
dtype0
�
Generator/dense/bias
VariableV2*
_output_shapes	
:�*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:�*
dtype0
�
Generator/dense/bias/AssignAssignGenerator/dense/bias&Generator/dense/bias/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(
�
Generator/dense/bias/readIdentityGenerator/dense/bias*
_output_shapes	
:�*
T0*'
_class
loc:@Generator/dense/bias
�
Generator/dense/MatMulMatMul)Generator/forth/Generator/forthleaky_reluGenerator/dense/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
Generator/dense/BiasAddBiasAddGenerator/dense/MatMulGenerator/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
b
Generator/TanhTanhGenerator/dense/BiasAdd*(
_output_shapes
:����������*
T0
w
Discriminator/realPlaceholder*
dtype0*(
_output_shapes
:����������*
shape:����������
�
^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/minConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HY��*
dtype0*
_output_shapes
: 
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HY�=*
dtype0*
_output_shapes
: 
�
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
��*

seed*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
seed2�
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/subSub\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/max\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
_output_shapes
: 
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mulMulfDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniform\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/sub*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
�
XDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniformAdd\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mul\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
�
=Discriminator/first/Discriminator/firstfully_connected/kernel
VariableV2*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignAssign=Discriminator/first/Discriminator/firstfully_connected/kernelXDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
BDiscriminator/first/Discriminator/firstfully_connected/kernel/readIdentity=Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
�
MDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
;Discriminator/first/Discriminator/firstfully_connected/bias
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container 
�
BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignAssign;Discriminator/first/Discriminator/firstfully_connected/biasMDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
@Discriminator/first/Discriminator/firstfully_connected/bias/readIdentity;Discriminator/first/Discriminator/firstfully_connected/bias*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:�
�
=Discriminator/first/Discriminator/firstfully_connected/MatMulMatMulDiscriminator/realBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
>Discriminator/first/Discriminator/firstfully_connected/BiasAddBiasAdd=Discriminator/first/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
|
7Discriminator/first/Discriminator/firstleaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
�
5Discriminator/first/Discriminator/firstleaky_relu/mulMul7Discriminator/first/Discriminator/firstleaky_relu/alpha>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
1Discriminator/first/Discriminator/firstleaky_reluMaximum5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/minConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *��=*
dtype0*
_output_shapes
: 
�
hDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
��*

seed*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
seed2�*
dtype0
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/subSub^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/max^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
_output_shapes
: 
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mulMulhDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniform^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/sub*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��
�
ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniformAdd^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mul^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��
�
?Discriminator/second/Discriminator/secondfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
��
�
FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignAssign?Discriminator/second/Discriminator/secondfully_connected/kernelZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
��*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(
�
DDiscriminator/second/Discriminator/secondfully_connected/kernel/readIdentity?Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
�
ODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
=Discriminator/second/Discriminator/secondfully_connected/bias
VariableV2*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignAssign=Discriminator/second/Discriminator/secondfully_connected/biasODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
BDiscriminator/second/Discriminator/secondfully_connected/bias/readIdentity=Discriminator/second/Discriminator/secondfully_connected/bias*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:�
�
?Discriminator/second/Discriminator/secondfully_connected/MatMulMatMul1Discriminator/first/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
@Discriminator/second/Discriminator/secondfully_connected/BiasAddBiasAdd?Discriminator/second/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
~
9Discriminator/second/Discriminator/secondleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
7Discriminator/second/Discriminator/secondleaky_relu/mulMul9Discriminator/second/Discriminator/secondleaky_relu/alpha@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
3Discriminator/second/Discriminator/secondleaky_reluMaximum7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
9Discriminator/out/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@Discriminator/out/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
7Discriminator/out/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv�*
dtype0*
_output_shapes
: 
�
7Discriminator/out/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 
�
ADiscriminator/out/kernel/Initializer/random_uniform/RandomUniformRandomUniform9Discriminator/out/kernel/Initializer/random_uniform/shape*
_output_shapes
:	�*

seed*
T0*+
_class!
loc:@Discriminator/out/kernel*
seed2�*
dtype0
�
7Discriminator/out/kernel/Initializer/random_uniform/subSub7Discriminator/out/kernel/Initializer/random_uniform/max7Discriminator/out/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
: 
�
7Discriminator/out/kernel/Initializer/random_uniform/mulMulADiscriminator/out/kernel/Initializer/random_uniform/RandomUniform7Discriminator/out/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�
�
3Discriminator/out/kernel/Initializer/random_uniformAdd7Discriminator/out/kernel/Initializer/random_uniform/mul7Discriminator/out/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�
�
Discriminator/out/kernel
VariableV2*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *+
_class!
loc:@Discriminator/out/kernel
�
Discriminator/out/kernel/AssignAssignDiscriminator/out/kernel3Discriminator/out/kernel/Initializer/random_uniform*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
Discriminator/out/kernel/readIdentityDiscriminator/out/kernel*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�
�
(Discriminator/out/bias/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
�
Discriminator/out/bias
VariableV2*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:*
dtype0
�
Discriminator/out/bias/AssignAssignDiscriminator/out/bias(Discriminator/out/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:
�
Discriminator/out/bias/readIdentityDiscriminator/out/bias*
T0*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:
�
Discriminator/out/MatMulMatMul3Discriminator/second/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Discriminator/out/BiasAddBiasAddDiscriminator/out/MatMulDiscriminator/out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
q
Discriminator/out/SigmoidSigmoidDiscriminator/out/BiasAdd*
T0*'
_output_shapes
:���������
�
?Discriminator/first_1/Discriminator/firstfully_connected/MatMulMatMulGenerator/TanhBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
@Discriminator/first_1/Discriminator/firstfully_connected/BiasAddBiasAdd?Discriminator/first_1/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
~
9Discriminator/first_1/Discriminator/firstleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
7Discriminator/first_1/Discriminator/firstleaky_relu/mulMul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
3Discriminator/first_1/Discriminator/firstleaky_reluMaximum7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
ADiscriminator/second_1/Discriminator/secondfully_connected/MatMulMatMul3Discriminator/first_1/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
BDiscriminator/second_1/Discriminator/secondfully_connected/BiasAddBiasAddADiscriminator/second_1/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
;Discriminator/second_1/Discriminator/secondleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
9Discriminator/second_1/Discriminator/secondleaky_relu/mulMul;Discriminator/second_1/Discriminator/secondleaky_relu/alphaBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
5Discriminator/second_1/Discriminator/secondleaky_reluMaximum9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Discriminator/out_1/MatMulMatMul5Discriminator/second_1/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Discriminator/out_1/BiasAddBiasAddDiscriminator/out_1/MatMulDiscriminator/out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
u
Discriminator/out_1/SigmoidSigmoidDiscriminator/out_1/BiasAdd*'
_output_shapes
:���������*
T0
W
LogLogDiscriminator/out/Sigmoid*
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
`
subSubsub/xDiscriminator/out_1/Sigmoid*
T0*'
_output_shapes
:���������
C
Log_1Logsub*
T0*'
_output_shapes
:���������
H
addAddLogLog_1*'
_output_shapes
:���������*
T0
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
NegNegMean*
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
d
sub_1Subsub_1/xDiscriminator/out_1/Sigmoid*'
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
Mean_1MeanLog_2Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
generator_loss/tagConst*
_output_shapes
: *
valueB Bgenerator_loss*
dtype0
_
generator_lossHistogramSummarygenerator_loss/tagMean_1*
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
N
gradients/Neg_grad/NegNeggradients/Fill*
_output_shapes
: *
T0
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
\
gradients/Mean_grad/ShapeShapeadd*
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
gradients/add_grad/SumSumgradients/Mean_grad/truediv(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
gradients/add_grad/Sum_1Sumgradients/Mean_grad/truediv*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
�
gradients/Log_grad/Reciprocal
ReciprocalDiscriminator/out/Sigmoid,^gradients/add_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
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
4gradients/Discriminator/out/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out/Sigmoidgradients/Log_grad/mul*
T0*'
_output_shapes
:���������
[
gradients/sub_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
u
gradients/sub_grad/Shape_1ShapeDiscriminator/out_1/Sigmoid*
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
gradients/sub_grad/SumSumgradients/Log_1_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
gradients/sub_grad/Sum_1Sumgradients/Log_1_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
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
4gradients/Discriminator/out/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
9gradients/Discriminator/out/BiasAdd_grad/tuple/group_depsNoOp5^gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad5^gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad
�
Agradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyIdentity4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������
�
Cgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid-gradients/sub_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
.gradients/Discriminator/out/MatMul_grad/MatMulMatMulAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
0gradients/Discriminator/out/MatMul_grad/MatMul_1MatMul3Discriminator/second/Discriminator/secondleaky_reluAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
�
8gradients/Discriminator/out/MatMul_grad/tuple/group_depsNoOp/^gradients/Discriminator/out/MatMul_grad/MatMul1^gradients/Discriminator/out/MatMul_grad/MatMul_1
�
@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyIdentity.gradients/Discriminator/out/MatMul_grad/MatMul9^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*A
_class7
53loc:@gradients/Discriminator/out/MatMul_grad/MatMul
�
Bgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Identity0gradients/Discriminator/out/MatMul_grad/MatMul_19^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
;gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad7^gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
�
Cgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
�
Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeShape7Discriminator/second/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Shape@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zerosFillJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Ogradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Xgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Igradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectSelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqual@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Kgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1SelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Fgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumSumIgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectXgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1SumKgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1Zgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Sgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeM^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1
�
[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeT^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*]
_classS
QOloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:����������*
T0
�
]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1T^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
0gradients/Discriminator/out_1/MatMul_grad/MatMulMatMulCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
2gradients/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
:gradients/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp1^gradients/Discriminator/out_1/MatMul_grad/MatMul3^gradients/Discriminator/out_1/MatMul_grad/MatMul_1
�
Bgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity0gradients/Discriminator/out_1/MatMul_grad/MatMul;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*C
_class9
75loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity2gradients/Discriminator/out_1/MatMul_grad/MatMul_1;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
gradients/AddNAddNCgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/MulMul[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumSumJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul9Discriminator/second/Discriminator/secondleaky_relu/alpha[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Pgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Wgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1
�
_gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Qgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Zgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Kgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Mgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Hgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumKgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectZgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeHgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumMgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1\gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Ugradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpM^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeO^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
�
]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeV^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1V^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_1AddNBgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	�*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
N
�
gradients/AddN_2AddN]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*
N
�
[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:�
�
`gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_2\^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
hgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_2a^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
jgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapePgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1`gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Rgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Ygradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeS^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
�
agradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeZ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1Z^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Ugradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Wgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul1Discriminator/first/Discriminator/firstleaky_reluhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
_gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
�
ggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
igradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_3AddN_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
T0*
data_formatNHWC*
_output_shapes	
:�
�
bgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_3^^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
jgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3c^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
�
lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradc^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeShape5Discriminator/first/Discriminator/firstleaky_relu/mul*
out_type0*
_output_shapes
:*
T0
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Shapeggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosFillHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Mgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Vgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Ggradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectSelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Igradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1SelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Dgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumSumGgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectVgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeReshapeDgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1SumIgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1Xgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Jgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Qgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpI^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeK^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1
�
Ygradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeR^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*[
_classQ
OMloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape
�
[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1R^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Wgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMuljgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Ygradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relujgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
agradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulZ^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
�
igradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulb^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1b^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_4AddNjgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
N
�
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
Zgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulMulYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumSumHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulZgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul7Discriminator/first/Discriminator/firstleaky_relu/alphaYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1\gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ngradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ugradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpM^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeO^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1
�
]gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeV^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1V^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapeigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Ogradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Xgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Igradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Kgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Fgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumIgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectXgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumKgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1Zgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Sgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeM^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeT^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1T^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
gradients/AddN_5AddNigradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
��*
T0
�
gradients/AddN_6AddN[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ygradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*
data_formatNHWC*
_output_shapes	
:�
�
^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6Z^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
fgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
hgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Wgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
�
_gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Sgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
Ugradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/realfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
]gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpT^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulV^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
egradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentitySgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
ggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityUgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
gradients/AddN_7AddN]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N
�
[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
_output_shapes	
:�*
T0*
data_formatNHWC
�
`gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_7\^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
hgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_7a^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*n
_classd
b`loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
Ugradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Wgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
_gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
ggradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*j
_class`
^\loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
gradients/AddN_8AddNhgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�
�
gradients/AddN_9AddNggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
��
�
beta1_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
�
beta1_power/readIdentitybeta1_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
�
beta2_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
beta2_power/readIdentitybeta2_power*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: *
T0
�
dDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0
�
ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
TDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zerosFilldDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
BDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam
VariableV2*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamTDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
�
GDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
�
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zerosFillfDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
DDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1
VariableV2*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignAssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/readIdentityDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
�
RDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zerosConst*
_output_shapes	
:�*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB�*    *
dtype0
�
@Discriminator/first/Discriminator/firstfully_connected/bias/Adam
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container 
�
GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignAssign@Discriminator/first/Discriminator/firstfully_connected/bias/AdamRDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
EDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/readIdentity@Discriminator/first/Discriminator/firstfully_connected/bias/Adam*
_output_shapes	
:�*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
BDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1
VariableV2*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:�*
dtype0
�
IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(
�
GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:�*
T0
�
fDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
VDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zerosFillfDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensor\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
DDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
�
KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamVDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
IDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��
�
hDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    
�
XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zerosFillhDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
FDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1
VariableV2*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
MDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/AssignAssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(
�
KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/readIdentityFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��*
T0
�
TDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
BDiscriminator/second/Discriminator/secondfully_connected/bias/Adam
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container 
�
IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/AssignAssignBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamTDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
GDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/readIdentityBDiscriminator/second/Discriminator/secondfully_connected/bias/Adam*
_output_shapes	
:�*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias
�
VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
DDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:�
�
KDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:�
�
/Discriminator/out/kernel/Adam/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
Discriminator/out/kernel/Adam
VariableV2*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
�
$Discriminator/out/kernel/Adam/AssignAssignDiscriminator/out/kernel/Adam/Discriminator/out/kernel/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	�
�
"Discriminator/out/kernel/Adam/readIdentityDiscriminator/out/kernel/Adam*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�
�
1Discriminator/out/kernel/Adam_1/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	�*    *
dtype0*
_output_shapes
:	�
�
Discriminator/out/kernel/Adam_1
VariableV2*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
�
&Discriminator/out/kernel/Adam_1/AssignAssignDiscriminator/out/kernel/Adam_11Discriminator/out/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	�*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(
�
$Discriminator/out/kernel/Adam_1/readIdentityDiscriminator/out/kernel/Adam_1*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�
�
-Discriminator/out/bias/Adam/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
�
Discriminator/out/bias/Adam
VariableV2*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:*
dtype0
�
"Discriminator/out/bias/Adam/AssignAssignDiscriminator/out/bias/Adam-Discriminator/out/bias/Adam/Initializer/zeros*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
 Discriminator/out/bias/Adam/readIdentityDiscriminator/out/bias/Adam*
T0*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:
�
/Discriminator/out/bias/Adam_1/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
�
Discriminator/out/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:
�
$Discriminator/out/bias/Adam_1/AssignAssignDiscriminator/out/bias/Adam_1/Discriminator/out/bias/Adam_1/Initializer/zeros*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
"Discriminator/out/bias/Adam_1/readIdentityDiscriminator/out/bias/Adam_1*
T0*)
_class
loc:@Discriminator/out/bias*
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
Adam/epsilonConst*
_output_shapes
: *
valueB
 *w�+2*
dtype0
�
SAdam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam	ApplyAdam=Discriminator/first/Discriminator/firstfully_connected/kernelBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
use_locking( *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
QAdam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdam	ApplyAdam;Discriminator/first/Discriminator/firstfully_connected/bias@Discriminator/first/Discriminator/firstfully_connected/bias/AdamBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
UAdam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam	ApplyAdam?Discriminator/second/Discriminator/secondfully_connected/kernelDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5* 
_output_shapes
:
��*
use_locking( *
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
use_nesterov( 
�
SAdam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdam	ApplyAdam=Discriminator/second/Discriminator/secondfully_connected/biasBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
use_locking( *
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
.Adam/update_Discriminator/out/kernel/ApplyAdam	ApplyAdamDiscriminator/out/kernelDiscriminator/out/kernel/AdamDiscriminator/out/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_locking( *
T0*+
_class!
loc:@Discriminator/out/kernel*
use_nesterov( *
_output_shapes
:	�
�
,Adam/update_Discriminator/out/bias/ApplyAdam	ApplyAdamDiscriminator/out/biasDiscriminator/out/bias/AdamDiscriminator/out/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
use_locking( *
T0*)
_class
loc:@Discriminator/out/bias*
use_nesterov( *
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
�
AdamNoOp^Adam/Assign^Adam/Assign_1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam
T
gradients_1/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Z
gradients_1/grad_ys_0Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*

index_type0*
_output_shapes
: *
T0
v
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
b
gradients_1/Mean_1_grad/ShapeShapeLog_2*
T0*
out_type0*
_output_shapes
:
�
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
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
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
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
Reciprocalsub_1 ^gradients_1/Mean_1_grad/truediv*
T0*'
_output_shapes
:���������
�
gradients_1/Log_2_grad/mulMulgradients_1/Mean_1_grad/truediv!gradients_1/Log_2_grad/Reciprocal*'
_output_shapes
:���������*
T0
_
gradients_1/sub_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
y
gradients_1/sub_1_grad/Shape_1ShapeDiscriminator/out_1/Sigmoid*
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
gradients_1/sub_1_grad/SumSumgradients_1/Log_2_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
�
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*
_output_shapes
: 
�
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid1gradients_1/sub_1_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
=gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp9^gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad9^gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
�
Egradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������
�
Ggradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*K
_classA
?=loc:@gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
2gradients_1/Discriminator/out_1/MatMul_grad/MatMulMatMulEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	�*
transpose_a(*
transpose_b( *
T0
�
<gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp3^gradients_1/Discriminator/out_1/MatMul_grad/MatMul5^gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1
�
Dgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/Discriminator/out_1/MatMul_grad/MatMul=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Fgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
out_type0*
_output_shapes
:*
T0
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Sgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Mgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Ogradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Jgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumMgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeJgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumOgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Wgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpO^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeQ^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
�
_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeX^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
agradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1X^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1bgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Tgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapePgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
[gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpS^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeU^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
�
cgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*g
_class]
[Yloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
�
gradients_1/AddNAddNagradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes	
:�
�
dgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN`^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
lgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddNe^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
ngradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrade^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*r
_classh
fdloc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
Ygradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMullgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
�
[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relulgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
cgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpZ^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul\^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
�
kgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityYgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMuld^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*l
_classb
`^loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul
�
mgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1Identity[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1d^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapekgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Qgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Zgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Kgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualkgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Mgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeroskgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Hgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumKgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectZgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeHgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumMgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1\gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ugradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpM^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeO^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeV^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1V^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapePgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1`gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Rgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ygradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeS^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
�
agradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeZ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*c
_classY
WUloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
�
cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1Z^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_1AddN_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:�
�
bgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1^^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
jgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1c^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
lgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradc^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Wgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMuljgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Ygradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhjgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
�
agradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulZ^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
igradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulb^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
kgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1b^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*l
_classb
`^loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
(gradients_1/Generator/Tanh_grad/TanhGradTanhGradGenerator/Tanhigradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/Generator/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:�
�
9gradients_1/Generator/dense/BiasAdd_grad/tuple/group_depsNoOp)^gradients_1/Generator/Tanh_grad/TanhGrad5^gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad
�
Agradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/Generator/Tanh_grad/TanhGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/Generator/Tanh_grad/TanhGrad*(
_output_shapes
:����������
�
Cgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
.gradients_1/Generator/dense/MatMul_grad/MatMulMatMulAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyGenerator/dense/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
0gradients_1/Generator/dense/MatMul_grad/MatMul_1MatMul)Generator/forth/Generator/forthleaky_reluAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
8gradients_1/Generator/dense/MatMul_grad/tuple/group_depsNoOp/^gradients_1/Generator/dense/MatMul_grad/MatMul1^gradients_1/Generator/dense/MatMul_grad/MatMul_1
�
@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/Generator/dense/MatMul_grad/MatMul9^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*A
_class7
53loc:@gradients_1/Generator/dense/MatMul_grad/MatMul
�
Bgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/Generator/dense/MatMul_grad/MatMul_19^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*C
_class9
75loc:@gradients_1/Generator/dense/MatMul_grad/MatMul_1
�
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeShape-Generator/forth/Generator/forthleaky_relu/mul*
out_type0*
_output_shapes
:*
T0
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Shape@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zerosFillBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Ggradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqualGreaterEqual-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Pgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Agradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectSelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Cgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1SelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/SumSumAgradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectPgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeReshape>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1SumCgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1Rgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Kgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeE^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeL^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*U
_classK
IGloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape*(
_output_shapes
:����������*
T0
�
Ugradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1L^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
Tgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulMulSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumSumBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulTgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Mul/Generator/forth/Generator/forthleaky_relu/alphaSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Hgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1
�
Wgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_2AddNUgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*
N
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_2fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_2hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
Tshape0*
_output_shapes	
:�*
T0
�
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/forth/Generator/forthfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAddigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape
�
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg
�
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Sgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Xgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad
�
`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape
�
bgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
_output_shapes	
:�*
T0
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/forth/Generator/forthbatch_normalized/moving_mean/read*
_output_shapes	
:�*
T0
�
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
�
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Mgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/forth/Generator/forthfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Ogradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1MatMul)Generator/third/Generator/thirdleaky_relu`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Wgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1
�
_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
agradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*b
_classX
VTloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1
�
gradients_1/AddN_3AddNkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*
_output_shapes	
:�*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_3:Generator/forth/Generator/forthbatch_normalized/gamma/read*
T0*
_output_shapes	
:�
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_3?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt*
_output_shapes	
:�*
T0
�
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1
�
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�*
T0
�
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeShape-Generator/third/Generator/thirdleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Shape_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zerosFillBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Ggradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqualGreaterEqual-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Pgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Agradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectSelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Cgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1SelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/SumSumAgradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectPgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeReshape>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1SumCgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1Rgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Kgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeE^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeL^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Ugradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1L^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Tgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulMulSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumSumBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulTgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Mul/Generator/third/Generator/thirdleaky_relu/alphaSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Hgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1
�
Wgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_4AddNUgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_4fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/third/Generator/thirdfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*(
_output_shapes
:����������*
T0
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAddigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape
�
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg
�
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Sgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
_output_shapes	
:�*
T0*
data_formatNHWC
�
Xgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad
�
`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
bgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:�
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/third/Generator/thirdbatch_normalized/moving_mean/read*
_output_shapes	
:�*
T0
�
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�*
T0
�
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Mgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/third/Generator/thirdfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Ogradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1MatMul+Generator/second/Generator/secondleaky_relu`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Wgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1
�
_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*`
_classV
TRloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul
�
agradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_5AddNkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_5:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:�
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_5?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1
�
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeShape/Generator/second/Generator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
�
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Shape_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/zerosFillDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Igradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualGreaterEqual/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Rgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Cgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectSelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqual_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Egradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1SelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumSumCgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectRgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeReshape@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1SumEgradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1Tgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Fgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1ReshapeBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Mgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_depsNoOpE^gradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeG^gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1
�
Ugradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyIdentityDgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeN^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape
�
Wgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1N^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Vgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulMulUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumSumDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulVgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeReshapeDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Mul1Generator/second/Generator/secondleaky_relu/alphaUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Xgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Jgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1ReshapeFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Qgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_depsNoOpI^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeK^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1
�
Ygradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeR^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
�
[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1R^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*]
_classS
QOloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1
�
gradients_1/AddN_6AddNWgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*
N
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_6hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
Tshape0*
_output_shapes	
:�*
T0
�
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape
�
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeShape8Generator/second/Generator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:�*
dtype0*
_output_shapes
:
�
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*(
_output_shapes
:����������*
T0
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumSumVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mulhgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAddkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�
�
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/NegNegmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
�
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpn^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1U^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg
�
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitymgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Negb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:�
�
Ugradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
_output_shapes	
:�*
T0*
data_formatNHWC
�
Zgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOpl^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyV^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
bgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitykgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
dgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
_output_shapes	
:�*
T0
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1Mulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1BGenerator/second/Generator/secondbatch_normalized/moving_mean/read*
T0*
_output_shapes	
:�
�
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulY^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Muld^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul
�
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Ogradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulMatMulbgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency<Generator/second/Generator/secondfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Qgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1MatMul)Generator/first/Generator/firstleaky_relubgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��*
transpose_a(*
transpose_b( 
�
Ygradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpP^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulR^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1
�
agradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulZ^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*b
_classX
VTloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul
�
cgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1Z^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*d
_classZ
XVloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1
�
gradients_1/AddN_7AddNmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*
_output_shapes	
:�*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_7<Generator/second/Generator/secondbatch_normalized/gamma/read*
T0*
_output_shapes	
:�
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_7AGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpU^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1
�
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mulb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:�
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeShape-Generator/first/Generator/firstleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Shapeagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosFillBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Ggradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualGreaterEqual-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Pgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Agradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectSelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Cgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1SelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqual@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
>gradients_1/Generator/first/Generator/firstleaky_relu_grad/SumSumAgradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectPgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeReshape>gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1SumCgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1Rgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Dgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Kgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeE^gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeL^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Ugradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1L^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Tgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulMulSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumSumBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulTgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Mul/Generator/first/Generator/firstleaky_relu/alphaSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Hgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1
�
Wgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape
�
Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_8AddNUgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Sgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Xgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_8T^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8Y^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
bgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Mgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/first/Generator/firstfully_connected/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(
�
Ogradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	d�*
transpose_a(
�
Wgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1
�
_gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������d*
T0*`
_classV
TRloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul
�
agradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
_output_shapes
:	d�*
T0*b
_classX
VTloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1
�
beta1_power_1/initial_valueConst*
_output_shapes
: *'
_class
loc:@Generator/dense/bias*
valueB
 *fff?*
dtype0
�
beta1_power_1
VariableV2*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@Generator/dense/bias
w
beta1_power_1/readIdentitybeta1_power_1*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
�
beta2_power_1/initial_valueConst*'
_class
loc:@Generator/dense/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power_1
VariableV2*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
w
beta2_power_1/readIdentitybeta2_power_1*
_output_shapes
: *
T0*'
_class
loc:@Generator/dense/bias
�
\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
�
RGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zerosFill\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	d�*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0
�
:Generator/first/Generator/firstfully_connected/kernel/Adam
VariableV2*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�
�
AGenerator/first/Generator/firstfully_connected/kernel/Adam/AssignAssign:Generator/first/Generator/firstfully_connected/kernel/AdamLGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(
�
?Generator/first/Generator/firstfully_connected/kernel/Adam/readIdentity:Generator/first/Generator/firstfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�
�
^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   �   *
dtype0
�
TGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	d�
�
<Generator/first/Generator/firstfully_connected/kernel/Adam_1
VariableV2*
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container 
�
CGenerator/first/Generator/firstfully_connected/kernel/Adam_1/AssignAssign<Generator/first/Generator/firstfully_connected/kernel/Adam_1NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
AGenerator/first/Generator/firstfully_connected/kernel/Adam_1/readIdentity<Generator/first/Generator/firstfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�
�
JGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
8Generator/first/Generator/firstfully_connected/bias/Adam
VariableV2*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
?Generator/first/Generator/firstfully_connected/bias/Adam/AssignAssign8Generator/first/Generator/firstfully_connected/bias/AdamJGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
=Generator/first/Generator/firstfully_connected/bias/Adam/readIdentity8Generator/first/Generator/firstfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:�
�
LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
:Generator/first/Generator/firstfully_connected/bias/Adam_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container 
�
AGenerator/first/Generator/firstfully_connected/bias/Adam_1/AssignAssign:Generator/first/Generator/firstfully_connected/bias/Adam_1LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
?Generator/first/Generator/firstfully_connected/bias/Adam_1/readIdentity:Generator/first/Generator/firstfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:�
�
^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
TGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zerosFill^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorTGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0
�
<Generator/second/Generator/secondfully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
��
�
CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignAssign<Generator/second/Generator/secondfully_connected/kernel/AdamNGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
AGenerator/second/Generator/secondfully_connected/kernel/Adam/readIdentity<Generator/second/Generator/secondfully_connected/kernel/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
�
`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0
�
VGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zerosFill`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorVGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
>Generator/second/Generator/secondfully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
��
�
EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignAssign>Generator/second/Generator/secondfully_connected/kernel/Adam_1PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
CGenerator/second/Generator/secondfully_connected/kernel/Adam_1/readIdentity>Generator/second/Generator/secondfully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
�
LGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB�*    
�
:Generator/second/Generator/secondfully_connected/bias/Adam
VariableV2*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAssign:Generator/second/Generator/secondfully_connected/bias/AdamLGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
?Generator/second/Generator/secondfully_connected/bias/Adam/readIdentity:Generator/second/Generator/secondfully_connected/bias/Adam*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:�
�
NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB�*    *
dtype0
�
<Generator/second/Generator/secondfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:�
�
CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignAssign<Generator/second/Generator/secondfully_connected/bias/Adam_1NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
AGenerator/second/Generator/secondfully_connected/bias/Adam_1/readIdentity<Generator/second/Generator/secondfully_connected/bias/Adam_1*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
�
NGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
<Generator/second/Generator/secondbatch_normalized/gamma/Adam
VariableV2*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam/AssignAssign<Generator/second/Generator/secondbatch_normalized/gamma/AdamNGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
AGenerator/second/Generator/secondbatch_normalized/gamma/Adam/readIdentity<Generator/second/Generator/secondbatch_normalized/gamma/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:�
�
PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container 
�
EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignAssign>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/readIdentity>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1*
_output_shapes	
:�*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma
�
MGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
;Generator/second/Generator/secondbatch_normalized/beta/Adam
VariableV2*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignAssign;Generator/second/Generator/secondbatch_normalized/beta/AdamMGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
@Generator/second/Generator/secondbatch_normalized/beta/Adam/readIdentity;Generator/second/Generator/secondbatch_normalized/beta/Adam*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:�
�
OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
=Generator/second/Generator/secondbatch_normalized/beta/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:�
�
DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignAssign=Generator/second/Generator/secondbatch_normalized/beta/Adam_1OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
BGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/readIdentity=Generator/second/Generator/secondbatch_normalized/beta/Adam_1*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:�*
T0
�
\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
RGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zerosFill\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/Const*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
:Generator/third/Generator/thirdfully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
��
�
AGenerator/third/Generator/thirdfully_connected/kernel/Adam/AssignAssign:Generator/third/Generator/thirdfully_connected/kernel/AdamLGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
?Generator/third/Generator/thirdfully_connected/kernel/Adam/readIdentity:Generator/third/Generator/thirdfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��
�
^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
TGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    
�
NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/Const*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
<Generator/third/Generator/thirdfully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
��
�
CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignAssign<Generator/third/Generator/thirdfully_connected/kernel/Adam_1NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
AGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/readIdentity<Generator/third/Generator/thirdfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��
�
JGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zerosConst*
_output_shapes	
:�*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB�*    *
dtype0
�
8Generator/third/Generator/thirdfully_connected/bias/Adam
VariableV2*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
?Generator/third/Generator/thirdfully_connected/bias/Adam/AssignAssign8Generator/third/Generator/thirdfully_connected/bias/AdamJGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
=Generator/third/Generator/thirdfully_connected/bias/Adam/readIdentity8Generator/third/Generator/thirdfully_connected/bias/Adam*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
�
LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
:Generator/third/Generator/thirdfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:�
�
AGenerator/third/Generator/thirdfully_connected/bias/Adam_1/AssignAssign:Generator/third/Generator/thirdfully_connected/bias/Adam_1LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
?Generator/third/Generator/thirdfully_connected/bias/Adam_1/readIdentity:Generator/third/Generator/thirdfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:�
�
LGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zerosConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
:Generator/third/Generator/thirdbatch_normalized/gamma/Adam
VariableV2*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAssign:Generator/third/Generator/thirdbatch_normalized/gamma/AdamLGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
�
?Generator/third/Generator/thirdbatch_normalized/gamma/Adam/readIdentity:Generator/third/Generator/thirdbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:�
�
NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB�*    *
dtype0*
_output_shapes	
:�
�
<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:�
�
CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignAssign<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/readIdentity<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:�
�
KGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
9Generator/third/Generator/thirdbatch_normalized/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:�
�
@Generator/third/Generator/thirdbatch_normalized/beta/Adam/AssignAssign9Generator/third/Generator/thirdbatch_normalized/beta/AdamKGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
>Generator/third/Generator/thirdbatch_normalized/beta/Adam/readIdentity9Generator/third/Generator/thirdbatch_normalized/beta/Adam*
_output_shapes	
:�*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
�
MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1
VariableV2*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignAssign;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
@Generator/third/Generator/thirdbatch_normalized/beta/Adam_1/readIdentity;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1*
_output_shapes	
:�*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
�
\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0
�
RGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0
�
LGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
:Generator/forth/Generator/forthfully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container *
shape:
��
�
AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAssign:Generator/forth/Generator/forthfully_connected/kernel/AdamLGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(
�
?Generator/forth/Generator/forthfully_connected/kernel/Adam/readIdentity:Generator/forth/Generator/forthfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��
�
^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
TGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0* 
_output_shapes
:
��
�
<Generator/forth/Generator/forthfully_connected/kernel/Adam_1
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container 
�
CGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/AssignAssign<Generator/forth/Generator/forthfully_connected/kernel/Adam_1NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
�
AGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/readIdentity<Generator/forth/Generator/forthfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��
�
ZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
PGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
JGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zerosFillZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorPGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:�
�
8Generator/forth/Generator/forthfully_connected/bias/Adam
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
�
?Generator/forth/Generator/forthfully_connected/bias/Adam/AssignAssign8Generator/forth/Generator/forthfully_connected/bias/AdamJGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
=Generator/forth/Generator/forthfully_connected/bias/Adam/readIdentity8Generator/forth/Generator/forthfully_connected/bias/Adam*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
�
\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�*
dtype0*
_output_shapes
:
�
RGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:�
�
:Generator/forth/Generator/forthfully_connected/bias/Adam_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
�
AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAssign:Generator/forth/Generator/forthfully_connected/bias/Adam_1LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
?Generator/forth/Generator/forthfully_connected/bias/Adam_1/readIdentity:Generator/forth/Generator/forthfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:�
�
\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:�*
dtype0*
_output_shapes
:
�
RGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    
�
LGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zerosFill\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:�
�
:Generator/forth/Generator/forthbatch_normalized/gamma/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:�
�
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAssign:Generator/forth/Generator/forthbatch_normalized/gamma/AdamLGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
?Generator/forth/Generator/forthbatch_normalized/gamma/Adam/readIdentity:Generator/forth/Generator/forthbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:�
�
^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:�*
dtype0*
_output_shapes
:
�
TGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
�
NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/Const*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0
�
<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container 
�
CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignAssign<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/readIdentity<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:�
�
[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:�*
dtype0*
_output_shapes
:
�
QGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0
�
KGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zerosFill[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorQGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:�
�
9Generator/forth/Generator/forthbatch_normalized/beta/Adam
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
�
@Generator/forth/Generator/forthbatch_normalized/beta/Adam/AssignAssign9Generator/forth/Generator/forthbatch_normalized/beta/AdamKGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
>Generator/forth/Generator/forthbatch_normalized/beta/Adam/readIdentity9Generator/forth/Generator/forthbatch_normalized/beta/Adam*
_output_shapes	
:�*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
�
]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:�*
dtype0*
_output_shapes
:
�
SGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
�
MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/Const*
_output_shapes	
:�*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0
�
;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1
VariableV2*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignAssign;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
@Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/readIdentity;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:�
�
=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0
�
3Generator/dense/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
-Generator/dense/kernel/Adam/Initializer/zerosFill=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3Generator/dense/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@Generator/dense/kernel*

index_type0* 
_output_shapes
:
��
�
Generator/dense/kernel/Adam
VariableV2*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
"Generator/dense/kernel/Adam/AssignAssignGenerator/dense/kernel/Adam-Generator/dense/kernel/Adam/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(
�
 Generator/dense/kernel/Adam/readIdentityGenerator/dense/kernel/Adam* 
_output_shapes
:
��*
T0*)
_class
loc:@Generator/dense/kernel
�
?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*)
_class
loc:@Generator/dense/kernel*
valueB"     
�
5Generator/dense/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/Generator/dense/kernel/Adam_1/Initializer/zerosFill?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5Generator/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@Generator/dense/kernel*

index_type0* 
_output_shapes
:
��
�
Generator/dense/kernel/Adam_1
VariableV2*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
$Generator/dense/kernel/Adam_1/AssignAssignGenerator/dense/kernel/Adam_1/Generator/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
��
�
"Generator/dense/kernel/Adam_1/readIdentityGenerator/dense/kernel/Adam_1*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
��*
T0
�
+Generator/dense/bias/Adam/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Generator/dense/bias/Adam
VariableV2*
_output_shapes	
:�*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:�*
dtype0
�
 Generator/dense/bias/Adam/AssignAssignGenerator/dense/bias/Adam+Generator/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:�
�
Generator/dense/bias/Adam/readIdentityGenerator/dense/bias/Adam*
_output_shapes	
:�*
T0*'
_class
loc:@Generator/dense/bias
�
-Generator/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*'
_class
loc:@Generator/dense/bias*
valueB�*    *
dtype0
�
Generator/dense/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
"Generator/dense/bias/Adam_1/AssignAssignGenerator/dense/bias/Adam_1-Generator/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:�
�
 Generator/dense/bias/Adam_1/readIdentityGenerator/dense/bias/Adam_1*
T0*'
_class
loc:@Generator/dense/bias*
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
MAdam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/first/Generator/firstfully_connected/kernel:Generator/first/Generator/firstfully_connected/kernel/Adam<Generator/first/Generator/firstfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
use_nesterov( *
_output_shapes
:	d�
�
KAdam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdam	ApplyAdam3Generator/first/Generator/firstfully_connected/bias8Generator/first/Generator/firstfully_connected/bias/Adam:Generator/first/Generator/firstfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
�
OAdam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdam	ApplyAdam7Generator/second/Generator/secondfully_connected/kernel<Generator/second/Generator/secondfully_connected/kernel/Adam>Generator/second/Generator/secondfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
MAdam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdam	ApplyAdam5Generator/second/Generator/secondfully_connected/bias:Generator/second/Generator/secondfully_connected/bias/Adam<Generator/second/Generator/secondfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
OAdam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdam	ApplyAdam7Generator/second/Generator/secondbatch_normalized/gamma<Generator/second/Generator/secondbatch_normalized/gamma/Adam>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
NAdam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdam	ApplyAdam6Generator/second/Generator/secondbatch_normalized/beta;Generator/second/Generator/secondbatch_normalized/beta/Adam=Generator/second/Generator/secondbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes	
:�*
use_locking( *
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
use_nesterov( 
�
MAdam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdfully_connected/kernel:Generator/third/Generator/thirdfully_connected/kernel/Adam<Generator/third/Generator/thirdfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0
�
KAdam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdam	ApplyAdam3Generator/third/Generator/thirdfully_connected/bias8Generator/third/Generator/thirdfully_connected/bias/Adam:Generator/third/Generator/thirdfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
MAdam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdbatch_normalized/gamma:Generator/third/Generator/thirdbatch_normalized/gamma/Adam<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:�
�
LAdam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/third/Generator/thirdbatch_normalized/beta9Generator/third/Generator/thirdbatch_normalized/beta/Adam;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
MAdam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthfully_connected/kernel:Generator/forth/Generator/forthfully_connected/kernel/Adam<Generator/forth/Generator/forthfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
�
KAdam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdam	ApplyAdam3Generator/forth/Generator/forthfully_connected/bias8Generator/forth/Generator/forthfully_connected/bias/Adam:Generator/forth/Generator/forthfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
MAdam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthbatch_normalized/gamma:Generator/forth/Generator/forthbatch_normalized/gamma/Adam<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:�
�
LAdam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/forth/Generator/forthbatch_normalized/beta9Generator/forth/Generator/forthbatch_normalized/beta/Adam;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
.Adam_1/update_Generator/dense/kernel/ApplyAdam	ApplyAdamGenerator/dense/kernelGenerator/dense/kernel/AdamGenerator/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1*)
_class
loc:@Generator/dense/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0
�
,Adam_1/update_Generator/dense/bias/ApplyAdam	ApplyAdamGenerator/dense/biasGenerator/dense/bias/AdamGenerator/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
use_nesterov( *
_output_shapes	
:�
�


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*'
_class
loc:@Generator/dense/bias
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
�

Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
�	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: ""7
	summaries*
(
discriminator_loss:0
generator_loss:0"�*
trainable_variables�*�*
�
7Generator/first/Generator/firstfully_connected/kernel:0<Generator/first/Generator/firstfully_connected/kernel/Assign<Generator/first/Generator/firstfully_connected/kernel/read:02RGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform:08
�
5Generator/first/Generator/firstfully_connected/bias:0:Generator/first/Generator/firstfully_connected/bias/Assign:Generator/first/Generator/firstfully_connected/bias/read:02GGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros:08
�
9Generator/second/Generator/secondfully_connected/kernel:0>Generator/second/Generator/secondfully_connected/kernel/Assign>Generator/second/Generator/secondfully_connected/kernel/read:02TGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform:08
�
7Generator/second/Generator/secondfully_connected/bias:0<Generator/second/Generator/secondfully_connected/bias/Assign<Generator/second/Generator/secondfully_connected/bias/read:02IGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros:08
�
9Generator/second/Generator/secondbatch_normalized/gamma:0>Generator/second/Generator/secondbatch_normalized/gamma/Assign>Generator/second/Generator/secondbatch_normalized/gamma/read:02JGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones:08
�
8Generator/second/Generator/secondbatch_normalized/beta:0=Generator/second/Generator/secondbatch_normalized/beta/Assign=Generator/second/Generator/secondbatch_normalized/beta/read:02JGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros:08
�
7Generator/third/Generator/thirdfully_connected/kernel:0<Generator/third/Generator/thirdfully_connected/kernel/Assign<Generator/third/Generator/thirdfully_connected/kernel/read:02RGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform:08
�
5Generator/third/Generator/thirdfully_connected/bias:0:Generator/third/Generator/thirdfully_connected/bias/Assign:Generator/third/Generator/thirdfully_connected/bias/read:02GGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros:08
�
7Generator/third/Generator/thirdbatch_normalized/gamma:0<Generator/third/Generator/thirdbatch_normalized/gamma/Assign<Generator/third/Generator/thirdbatch_normalized/gamma/read:02HGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones:08
�
6Generator/third/Generator/thirdbatch_normalized/beta:0;Generator/third/Generator/thirdbatch_normalized/beta/Assign;Generator/third/Generator/thirdbatch_normalized/beta/read:02HGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros:08
�
7Generator/forth/Generator/forthfully_connected/kernel:0<Generator/forth/Generator/forthfully_connected/kernel/Assign<Generator/forth/Generator/forthfully_connected/kernel/read:02RGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform:08
�
5Generator/forth/Generator/forthfully_connected/bias:0:Generator/forth/Generator/forthfully_connected/bias/Assign:Generator/forth/Generator/forthfully_connected/bias/read:02GGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros:08
�
7Generator/forth/Generator/forthbatch_normalized/gamma:0<Generator/forth/Generator/forthbatch_normalized/gamma/Assign<Generator/forth/Generator/forthbatch_normalized/gamma/read:02HGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones:08
�
6Generator/forth/Generator/forthbatch_normalized/beta:0;Generator/forth/Generator/forthbatch_normalized/beta/Assign;Generator/forth/Generator/forthbatch_normalized/beta/read:02HGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros:08
�
Generator/dense/kernel:0Generator/dense/kernel/AssignGenerator/dense/kernel/read:023Generator/dense/kernel/Initializer/random_uniform:08
~
Generator/dense/bias:0Generator/dense/bias/AssignGenerator/dense/bias/read:02(Generator/dense/bias/Initializer/zeros:08
�
?Discriminator/first/Discriminator/firstfully_connected/kernel:0DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/read:02ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform:08
�
=Discriminator/first/Discriminator/firstfully_connected/bias:0BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignBDiscriminator/first/Discriminator/firstfully_connected/bias/read:02ODiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros:08
�
ADiscriminator/second/Discriminator/secondfully_connected/kernel:0FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/read:02\Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform:08
�
?Discriminator/second/Discriminator/secondfully_connected/bias:0DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignDDiscriminator/second/Discriminator/secondfully_connected/bias/read:02QDiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros:08
�
Discriminator/out/kernel:0Discriminator/out/kernel/AssignDiscriminator/out/kernel/read:025Discriminator/out/kernel/Initializer/random_uniform:08
�
Discriminator/out/bias:0Discriminator/out/bias/AssignDiscriminator/out/bias/read:02*Discriminator/out/bias/Initializer/zeros:08"
train_op

Adam
Adam_1"��
	variables����
�
7Generator/first/Generator/firstfully_connected/kernel:0<Generator/first/Generator/firstfully_connected/kernel/Assign<Generator/first/Generator/firstfully_connected/kernel/read:02RGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform:08
�
5Generator/first/Generator/firstfully_connected/bias:0:Generator/first/Generator/firstfully_connected/bias/Assign:Generator/first/Generator/firstfully_connected/bias/read:02GGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros:08
�
9Generator/second/Generator/secondfully_connected/kernel:0>Generator/second/Generator/secondfully_connected/kernel/Assign>Generator/second/Generator/secondfully_connected/kernel/read:02TGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform:08
�
7Generator/second/Generator/secondfully_connected/bias:0<Generator/second/Generator/secondfully_connected/bias/Assign<Generator/second/Generator/secondfully_connected/bias/read:02IGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros:08
�
9Generator/second/Generator/secondbatch_normalized/gamma:0>Generator/second/Generator/secondbatch_normalized/gamma/Assign>Generator/second/Generator/secondbatch_normalized/gamma/read:02JGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones:08
�
8Generator/second/Generator/secondbatch_normalized/beta:0=Generator/second/Generator/secondbatch_normalized/beta/Assign=Generator/second/Generator/secondbatch_normalized/beta/read:02JGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros:08
�
?Generator/second/Generator/secondbatch_normalized/moving_mean:0DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignDGenerator/second/Generator/secondbatch_normalized/moving_mean/read:02QGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros:0
�
CGenerator/second/Generator/secondbatch_normalized/moving_variance:0HGenerator/second/Generator/secondbatch_normalized/moving_variance/AssignHGenerator/second/Generator/secondbatch_normalized/moving_variance/read:02TGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/ones:0
�
7Generator/third/Generator/thirdfully_connected/kernel:0<Generator/third/Generator/thirdfully_connected/kernel/Assign<Generator/third/Generator/thirdfully_connected/kernel/read:02RGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform:08
�
5Generator/third/Generator/thirdfully_connected/bias:0:Generator/third/Generator/thirdfully_connected/bias/Assign:Generator/third/Generator/thirdfully_connected/bias/read:02GGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros:08
�
7Generator/third/Generator/thirdbatch_normalized/gamma:0<Generator/third/Generator/thirdbatch_normalized/gamma/Assign<Generator/third/Generator/thirdbatch_normalized/gamma/read:02HGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones:08
�
6Generator/third/Generator/thirdbatch_normalized/beta:0;Generator/third/Generator/thirdbatch_normalized/beta/Assign;Generator/third/Generator/thirdbatch_normalized/beta/read:02HGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros:08
�
=Generator/third/Generator/thirdbatch_normalized/moving_mean:0BGenerator/third/Generator/thirdbatch_normalized/moving_mean/AssignBGenerator/third/Generator/thirdbatch_normalized/moving_mean/read:02OGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zeros:0
�
AGenerator/third/Generator/thirdbatch_normalized/moving_variance:0FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignFGenerator/third/Generator/thirdbatch_normalized/moving_variance/read:02RGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones:0
�
7Generator/forth/Generator/forthfully_connected/kernel:0<Generator/forth/Generator/forthfully_connected/kernel/Assign<Generator/forth/Generator/forthfully_connected/kernel/read:02RGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform:08
�
5Generator/forth/Generator/forthfully_connected/bias:0:Generator/forth/Generator/forthfully_connected/bias/Assign:Generator/forth/Generator/forthfully_connected/bias/read:02GGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros:08
�
7Generator/forth/Generator/forthbatch_normalized/gamma:0<Generator/forth/Generator/forthbatch_normalized/gamma/Assign<Generator/forth/Generator/forthbatch_normalized/gamma/read:02HGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones:08
�
6Generator/forth/Generator/forthbatch_normalized/beta:0;Generator/forth/Generator/forthbatch_normalized/beta/Assign;Generator/forth/Generator/forthbatch_normalized/beta/read:02HGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros:08
�
=Generator/forth/Generator/forthbatch_normalized/moving_mean:0BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignBGenerator/forth/Generator/forthbatch_normalized/moving_mean/read:02OGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros:0
�
AGenerator/forth/Generator/forthbatch_normalized/moving_variance:0FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignFGenerator/forth/Generator/forthbatch_normalized/moving_variance/read:02RGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones:0
�
Generator/dense/kernel:0Generator/dense/kernel/AssignGenerator/dense/kernel/read:023Generator/dense/kernel/Initializer/random_uniform:08
~
Generator/dense/bias:0Generator/dense/bias/AssignGenerator/dense/bias/read:02(Generator/dense/bias/Initializer/zeros:08
�
?Discriminator/first/Discriminator/firstfully_connected/kernel:0DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/read:02ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform:08
�
=Discriminator/first/Discriminator/firstfully_connected/bias:0BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignBDiscriminator/first/Discriminator/firstfully_connected/bias/read:02ODiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros:08
�
ADiscriminator/second/Discriminator/secondfully_connected/kernel:0FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/read:02\Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform:08
�
?Discriminator/second/Discriminator/secondfully_connected/bias:0DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignDDiscriminator/second/Discriminator/secondfully_connected/bias/read:02QDiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros:08
�
Discriminator/out/kernel:0Discriminator/out/kernel/AssignDiscriminator/out/kernel/read:025Discriminator/out/kernel/Initializer/random_uniform:08
�
Discriminator/out/bias:0Discriminator/out/bias/AssignDiscriminator/out/bias/read:02*Discriminator/out/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
�
DDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam:0IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/AssignIDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/read:02VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros:0
�
FDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1:0KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignKDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/read:02XDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros:0
�
BDiscriminator/first/Discriminator/firstfully_connected/bias/Adam:0GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignGDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/read:02TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros:0
�
DDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1:0IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignIDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/read:02VDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros:0
�
FDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam:0KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/AssignKDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/read:02XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros:0
�
HDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1:0MDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/AssignMDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/read:02ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros:0
�
DDiscriminator/second/Discriminator/secondfully_connected/bias/Adam:0IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/AssignIDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/read:02VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zeros:0
�
FDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1:0KDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/AssignKDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/read:02XDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zeros:0
�
Discriminator/out/kernel/Adam:0$Discriminator/out/kernel/Adam/Assign$Discriminator/out/kernel/Adam/read:021Discriminator/out/kernel/Adam/Initializer/zeros:0
�
!Discriminator/out/kernel/Adam_1:0&Discriminator/out/kernel/Adam_1/Assign&Discriminator/out/kernel/Adam_1/read:023Discriminator/out/kernel/Adam_1/Initializer/zeros:0
�
Discriminator/out/bias/Adam:0"Discriminator/out/bias/Adam/Assign"Discriminator/out/bias/Adam/read:02/Discriminator/out/bias/Adam/Initializer/zeros:0
�
Discriminator/out/bias/Adam_1:0$Discriminator/out/bias/Adam_1/Assign$Discriminator/out/bias/Adam_1/read:021Discriminator/out/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
�
<Generator/first/Generator/firstfully_connected/kernel/Adam:0AGenerator/first/Generator/firstfully_connected/kernel/Adam/AssignAGenerator/first/Generator/firstfully_connected/kernel/Adam/read:02NGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros:0
�
>Generator/first/Generator/firstfully_connected/kernel/Adam_1:0CGenerator/first/Generator/firstfully_connected/kernel/Adam_1/AssignCGenerator/first/Generator/firstfully_connected/kernel/Adam_1/read:02PGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros:0
�
:Generator/first/Generator/firstfully_connected/bias/Adam:0?Generator/first/Generator/firstfully_connected/bias/Adam/Assign?Generator/first/Generator/firstfully_connected/bias/Adam/read:02LGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros:0
�
<Generator/first/Generator/firstfully_connected/bias/Adam_1:0AGenerator/first/Generator/firstfully_connected/bias/Adam_1/AssignAGenerator/first/Generator/firstfully_connected/bias/Adam_1/read:02NGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zeros:0
�
>Generator/second/Generator/secondfully_connected/kernel/Adam:0CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignCGenerator/second/Generator/secondfully_connected/kernel/Adam/read:02PGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros:0
�
@Generator/second/Generator/secondfully_connected/kernel/Adam_1:0EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignEGenerator/second/Generator/secondfully_connected/kernel/Adam_1/read:02RGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros:0
�
<Generator/second/Generator/secondfully_connected/bias/Adam:0AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAGenerator/second/Generator/secondfully_connected/bias/Adam/read:02NGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros:0
�
>Generator/second/Generator/secondfully_connected/bias/Adam_1:0CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignCGenerator/second/Generator/secondfully_connected/bias/Adam_1/read:02PGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros:0
�
>Generator/second/Generator/secondbatch_normalized/gamma/Adam:0CGenerator/second/Generator/secondbatch_normalized/gamma/Adam/AssignCGenerator/second/Generator/secondbatch_normalized/gamma/Adam/read:02PGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zeros:0
�
@Generator/second/Generator/secondbatch_normalized/gamma/Adam_1:0EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignEGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/read:02RGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros:0
�
=Generator/second/Generator/secondbatch_normalized/beta/Adam:0BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignBGenerator/second/Generator/secondbatch_normalized/beta/Adam/read:02OGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros:0
�
?Generator/second/Generator/secondbatch_normalized/beta/Adam_1:0DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignDGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/read:02QGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros:0
�
<Generator/third/Generator/thirdfully_connected/kernel/Adam:0AGenerator/third/Generator/thirdfully_connected/kernel/Adam/AssignAGenerator/third/Generator/thirdfully_connected/kernel/Adam/read:02NGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros:0
�
>Generator/third/Generator/thirdfully_connected/kernel/Adam_1:0CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignCGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/read:02PGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros:0
�
:Generator/third/Generator/thirdfully_connected/bias/Adam:0?Generator/third/Generator/thirdfully_connected/bias/Adam/Assign?Generator/third/Generator/thirdfully_connected/bias/Adam/read:02LGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros:0
�
<Generator/third/Generator/thirdfully_connected/bias/Adam_1:0AGenerator/third/Generator/thirdfully_connected/bias/Adam_1/AssignAGenerator/third/Generator/thirdfully_connected/bias/Adam_1/read:02NGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zeros:0
�
<Generator/third/Generator/thirdbatch_normalized/gamma/Adam:0AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/read:02NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros:0
�
>Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1:0CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignCGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/read:02PGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros:0
�
;Generator/third/Generator/thirdbatch_normalized/beta/Adam:0@Generator/third/Generator/thirdbatch_normalized/beta/Adam/Assign@Generator/third/Generator/thirdbatch_normalized/beta/Adam/read:02MGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zeros:0
�
=Generator/third/Generator/thirdbatch_normalized/beta/Adam_1:0BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignBGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/read:02OGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros:0
�
<Generator/forth/Generator/forthfully_connected/kernel/Adam:0AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAGenerator/forth/Generator/forthfully_connected/kernel/Adam/read:02NGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros:0
�
>Generator/forth/Generator/forthfully_connected/kernel/Adam_1:0CGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/AssignCGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/read:02PGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros:0
�
:Generator/forth/Generator/forthfully_connected/bias/Adam:0?Generator/forth/Generator/forthfully_connected/bias/Adam/Assign?Generator/forth/Generator/forthfully_connected/bias/Adam/read:02LGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros:0
�
<Generator/forth/Generator/forthfully_connected/bias/Adam_1:0AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAGenerator/forth/Generator/forthfully_connected/bias/Adam_1/read:02NGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros:0
�
<Generator/forth/Generator/forthbatch_normalized/gamma/Adam:0AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/read:02NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros:0
�
>Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1:0CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignCGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/read:02PGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros:0
�
;Generator/forth/Generator/forthbatch_normalized/beta/Adam:0@Generator/forth/Generator/forthbatch_normalized/beta/Adam/Assign@Generator/forth/Generator/forthbatch_normalized/beta/Adam/read:02MGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros:0
�
=Generator/forth/Generator/forthbatch_normalized/beta/Adam_1:0BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignBGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/read:02OGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros:0
�
Generator/dense/kernel/Adam:0"Generator/dense/kernel/Adam/Assign"Generator/dense/kernel/Adam/read:02/Generator/dense/kernel/Adam/Initializer/zeros:0
�
Generator/dense/kernel/Adam_1:0$Generator/dense/kernel/Adam_1/Assign$Generator/dense/kernel/Adam_1/read:021Generator/dense/kernel/Adam_1/Initializer/zeros:0
�
Generator/dense/bias/Adam:0 Generator/dense/bias/Adam/Assign Generator/dense/bias/Adam/read:02-Generator/dense/bias/Adam/Initializer/zeros:0
�
Generator/dense/bias/Adam_1:0"Generator/dense/bias/Adam_1/Assign"Generator/dense/bias/Adam_1/read:02/Generator/dense/bias/Adam_1/Initializer/zeros:0�8��       �N�	S5_���A*�
w
discriminator_loss*a	    m#�?    m#�?      �?!    m#�?) H��� @23?��|�?�E̟���?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) @KG��?2uo�p�+Se*8��������:              �?        ����       �{�	�W_���A*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) B)��	@2S�Fi��?ܔ�.�u�?�������:              �?        
s
generator_loss*a	    S���    S���      �?!    S���) ���@2yL�������E̟�����������:              �?        �$�       �{�	�h^_���A
*�
w
discriminator_loss*a	   @�@   @�@      �?!   @�@) q���G@2w`<f@�6v��@�������:              �?        
s
generator_loss*a	   �"��   �"��      �?!   �"��) d��U}@2w`<f���tM��������:              �?        "P���       �{�	�=f_���A*�
w
discriminator_loss*a	   �"@@   �"@@      �?!   �"@@)@�J�J�@2ܔ�.�u�?��tM@�������:              �?        
s
generator_loss*a	   ��K �   ��K �      �?!   ��K �) !��ݘ@2��tM�ܔ�.�u���������:              �?        �Ƞ�       �{�	5?m_���A*�
w
discriminator_loss*a	   ����?   ����?      �?!   ����?) �؝�X�?2\l�9�?+Se*8�?�������:              �?        
s
generator_loss*a	   ��޿   ��޿      �?!   ��޿) �O�i��?2�1%���Z%�޿�������:              �?        �,,@�       �{�	oAt_���A*�
w
discriminator_loss*a	   �0��?   �0��?      �?!   �0��?) ��pԯ?2�Z�_���?����?�������:              �?        
s
generator_loss*a	   ��Ҩ�   ��Ҩ�      �?!   ��Ҩ�) 2	8^Ac?2���g�骿�g���w���������:              �?        Iy�!�       �{�	�3{_���A*�
w
discriminator_loss*a	   �wz�?   �wz�?      �?!   �wz�?) 1�~��?2�@�"��?�K?�?�������:              �?        
s
generator_loss*a	   �$���   �$���      �?!   �$���) ~ ֨I-?2�#�h/���7c_XY���������:              �?        -�FQ�       �{�	�V�_���A#*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)�P���ɏ?2!�����?Ӗ8��s�?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  ���C?2}Y�4j���"�uԖ��������:              �?        ����       �{�	���_���A(*�
w
discriminator_loss*a	   `_��?   `_��?      �?!   `_��?)@���lƕ?2Ӗ8��s�?�?>8s2�?�������:              �?        
s
generator_loss*a	   �����   �����      �?!   �����) ��c=�x?2� l(����{ �ǳ���������:              �?        pI�B�       �{�	���_���A-*�
w
discriminator_loss*a	   ��<�?   ��<�?      �?!   ��<�?) �5.L~�?2��(!�ؼ?!�����?�������:              �?        
s
generator_loss*a	    P��    P��      �?!    P��) ��գn?2����iH��I�����������:              �?        n���       �{�	��_���A2*�
w
discriminator_loss*a	   �4�?   �4�?      �?!   �4�?) �k��hp?2I���?����iH�?�������:              �?        
s
generator_loss*a	    _rx�    _rx�      �?!    _rx�) �E'�?2o��5sz�*QH�x��������:              �?        r�-�       �{�	��_���A7*�
w
discriminator_loss*a	   @�/�?   @�/�?      �?!   @�/�?)����en?2I���?����iH�?�������:              �?        
s
generator_loss*a	   �FL�   �FL�      �?!   �FL�)�l�%��>2IcD���L��qU���I��������:              �?        OZP�       �{�	d��_���A<*�
w
discriminator_loss*a	   ��[�?   ��[�?      �?!   ��[�?) �*Z��\?2`��a�8�?�/�*>�?�������:              �?        
s
generator_loss*a	    
�2�    
�2�      �?!    
�2�) @��`v>2�u�w74���82��������:              �?        -���       �{�	R�_���AA*�
w
discriminator_loss*a	   ��,�?   ��,�?      �?!   ��,�?) bJ���@?2�"�uԖ?}Y�4j�?�������:              �?        
s
generator_loss*a	   @�,�   @�,�      �?!   @�,�)�L�{�h>2��VlQ.��7Kaa+��������:              �?        ͳ0��       �{�	N�_���AF*�
w
discriminator_loss*a	    6�?    6�?      �?!    6�?) @��u�;?2^�S���?�"�uԖ?�������:              �?        
s
generator_loss*a	   ฌ(�   ฌ(�      �?!   ฌ(�) �%��b>2I�I�)�(�+A�F�&��������:              �?        �����       �{�	W��_���AK*�
w
discriminator_loss*a	    �3�?    �3�?      �?!    �3�?) ��x��(?2#�+(�ŉ?�7c_XY�?�������:              �?        
s
generator_loss*a	   �wu2�   �wu2�      �?!   �wu2�) �,G�Ku>2�u�w74���82��������:              �?        50��       �{�	���_���AP*�
w
discriminator_loss*a	   ���{?   ���{?      �?!   ���{?) D���j?2o��5sz?���T}?�������:              �?        
s
generator_loss*a	   �]�9�   �]�9�      �?!   �]�9�) d�Z�`�>2��%>��:�uܬ�@8��������:              �?        �c�*�       �{�	�A`���AU*�
w
discriminator_loss*a	   `l�?   `l�?      �?!   `l�?) 	E��?2���T}?>	� �?�������:              �?        
s
generator_loss*a	   �}F�   �}F�      �?!   �}F�) dX�]W�>2
����G�a�$��{E��������:              �?        En��       �{�	�o`���AZ*�
w
discriminator_loss*a	    z�?    z�?      �?!    z�?) ���?2���T}?>	� �?�������:              �?        
s
generator_loss*a	   �SO�   �SO�      �?!   �SO�) ��[	=�>2k�1^�sO�IcD���L��������:              �?        E���       �{�	ݘ`���A_*�
w
discriminator_loss*a	   �?sy?   �?sy?      �?!   �?sy?)� ���=?2*QH�x?o��5sz?�������:              �?        
s
generator_loss*a	    �V�    �V�      �?!    �V�) �ƣ�[�>2ܗ�SsW�<DKc��T��������:              �?         b��       �{�	=�`���Ad*�
w
discriminator_loss*a	    �=x?    �=x?      �?!    �=x?) ���\?2*QH�x?o��5sz?�������:              �?        
s
generator_loss*a	   @7ha�   @7ha�      �?!   @7ha�) ��6��>2���%��b��l�P�`��������:              �?        �Ư<�       �{�	�-#`���Ai*�
w
discriminator_loss*a	   ���y?   ���y?      �?!   ���y?)�|�i��?2*QH�x?o��5sz?�������:              �?        
s
generator_loss*a	    Np�    Np�      �?!    Np�) @|��0�>2;8�clp��N�W�m��������:              �?        ����       �{�	�R*`���An*�
w
discriminator_loss*a	   ��\�?   ��\�?      �?!   ��\�?) �l�?e'?2#�+(�ŉ?�7c_XY�?�������:              �?        
s
generator_loss*a	   ��P~�   ��P~�      �?!   ��P~�)���w�?2>	� �����T}��������:              �?        "�3��       �{�	b1`���As*�
w
discriminator_loss*a	   @�p�?   @�p�?      �?!   @�p�?) �0$�3?2���&�?�Rc�ݒ?�������:              �?        
s
generator_loss*a	   ��L��   ��L��      �?!   ��L��)�(,�՝%?2�7c_XY��#�+(�ŉ��������:              �?        �fA|�       �{�	��8`���Ax*�
w
discriminator_loss*a	   �(�?   �(�?      �?!   �(�?) ɟj,2?2�#�h/�?���&�?�������:              �?        
s
generator_loss*a	   �C���   �C���      �?!   �C���) ��F"�!?2#�+(�ŉ�eiS�m���������:              �?        ��!��       �{�	,h`���A}*�
w
discriminator_loss*a	   ��z�?   ��z�?      �?!   ��z�?)@?Ì6:?2�Rc�ݒ?^�S���?�������:              �?        
s
generator_loss*a	   ���x�   ���x�      �?!   ���x�) %��D?2o��5sz�*QH�x��������:              �?        T+Ӵ�       b�D�	�o`���A�*�
w
discriminator_loss*a	    ^M�?    ^M�?      �?!    ^M�?) @�2�0?2�#�h/�?���&�?�������:              �?        
s
generator_loss*a	   `�:f�   `�:f�      �?!   `�:f�)@ZqRn��>2Tw��Nof�5Ucv0ed��������:              �?        �}��       b�D�	��v`���A�*�
w
discriminator_loss*a	   �;�?   �;�?      �?!   �;�?) BZ��X"?2eiS�m�?#�+(�ŉ?�������:              �?        
s
generator_loss*a	   �|b�   �|b�      �?!   �|b�)@:M��Z�>2���%��b��l�P�`��������:              �?        �oV�       b�D�	�}`���A�*�
w
discriminator_loss*a	   �g�?   �g�?      �?!   �g�?)�`LВJ(?2#�+(�ŉ?�7c_XY�?�������:              �?        
s
generator_loss*a	   `�j�   `�j�      �?!   `�j�) 'Ұ)J�>2ߤ�(g%k�P}���h��������:              �?        ��C)�       b�D�	ǅ`���A�*�
w
discriminator_loss*a	   ಟ�?   ಟ�?      �?!   ಟ�?)@��r��5?2���&�?�Rc�ݒ?�������:              �?        
s
generator_loss*a	   @f}�   @f}�      �?!   @f}�)�����y
?2>	� �����T}��������:              �?        ����       b�D�	�*�`���A�*�
w
discriminator_loss*a	   ��F�?   ��F�?      �?!   ��F�?)���<*�@?2�"�uԖ?}Y�4j�?�������:              �?        
s
generator_loss*a	   ��D��   ��D��      �?!   ��D��) DU�d4?2���J�\������=����������:              �?        �c�A�       b�D�	��`���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �<T��F?2}Y�4j�?��<�A��?�������:              �?        
s
generator_loss*a	   �}��   �}��      �?!   �}��) �:]?2���J�\������=����������:              �?        R�"�       b�D�	t��`���A�*�
w
discriminator_loss*a	   �CĢ?   �CĢ?      �?!   �CĢ?) ����V?2�uS��a�?`��a�8�?�������:              �?        
s
generator_loss*a	    ����    ����      �?!    ����) ���;/@?2�"�uԖ�^�S�����������:              �?        _L͘�       b�D�	i�`���A�*�
w
discriminator_loss*a	   @(8�?   @(8�?      �?!   @(8�?) A%���?2����?_&A�o��?�������:              �?        
s
generator_loss*a	   `/�̿   `/�̿      �?!   `/�̿) #`/]��?2�Z�_��ο�K?̿�������:              �?        U'�       b�D�	�+�`���A�*�
w
discriminator_loss*a	    o�@    o�@      �?!    o�@) ���6J-@2u�rʭ�@�DK��@�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) �bgH�&@2u�rʭ���Š)U	��������:              �?        �7�       b�D�	vb�`���A�*�
w
discriminator_loss*a	    �o&@    �o&@      �?!    �o&@) ��m�u_@2sQ��"�%@e��:�(@�������:              �?        
s
generator_loss*a	   �5$�   �5$�      �?!   �5$�) ���+Y@2sQ��"�%��ՠ�M�#��������:              �?        Tsh��       b�D�	�c�`���A�*�
w
discriminator_loss*a	   ��@   ��@      �?!   ��@)����JI-@2u�rʭ�@�DK��@�������:              �?        
s
generator_loss*a	   ��   ��      �?!   ��) ����(@2�DK���u�rʭ���������:              �?        5n���       b�D�	7\�`���A�*�
w
discriminator_loss*a	    ?Ƽ?    ?Ƽ?      �?!    ?Ƽ?) l�߉?2%g�cE9�?��(!�ؼ?�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) P�"�=2��~��¾�[�=�k���������:              �?        ���V�       �E��	�H�`���A�*�
w
discriminator_loss*a	    )��?    )��?      �?!    )��?) �\��uc?2�g���w�?���g��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��F��       �E��	�a���A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)��-��F?2}Y�4j�?��<�A��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        |ɠj�       �E��	�a���A�*�
w
discriminator_loss*a	    h�?    h�?      �?!    h�?)@�]���1?2�#�h/�?���&�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���f�       �E��	
�Ga���A�*�
w
discriminator_loss*a	   `�ݑ?   `�ݑ?      �?!   `�ݑ?)@�����3?2���&�?�Rc�ݒ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        5~[^�       �E��	�LOa���A�*�
w
discriminator_loss*a	   �bh�?   �bh�?      �?!   �bh�?)@�@��?2���J�\�?-Ա�L�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        X�P|�       �E��	K�Va���A�*�
w
discriminator_loss*a	   �F�?   �F�?      �?!   �F�?) ��D?O?2���J�\�?-Ա�L�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �X��       �E��	Q�^a���A�*�
w
discriminator_loss*a	   `�*|?   `�*|?      �?!   `�*|?) ��S�?2o��5sz?���T}?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        vPl�       �E��	�	fa���A�*�
w
discriminator_loss*a	    Bۏ?    Bۏ?      �?!    Bۏ?)  �/��/?2�#�h/�?���&�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � ���       �E��	��la���A�*�
w
discriminator_loss*a	   �,lq?   �,lq?      �?!   �,lq?)@�M����>2;8�clp?uWy��r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #��-�       �E��	��sa���A�*�
w
discriminator_loss*a	    K;x?    K;x?      �?!    K;x?) ^��^Y?2*QH�x?o��5sz?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        2����       �E��	�){a���A�*�
w
discriminator_loss*a	   �Hn�?   �Hn�?      �?!   �Hn�?) �(&��?2>	� �?����=��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        -�d��       �E��	��a���A�*�
w
discriminator_loss*a	    �v?    �v?      �?!    �v?)  �U ?2&b՞
�u?*QH�x?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7���       �E��	q\�a���A�*�
w
discriminator_loss*a	   ��m?   ��m?      �?!   ��m?) ���"��>2ߤ�(g%k?�N�W�m?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        WL��       �E��	�a���A�*�
w
discriminator_loss*a	    �=v?    �=v?      �?!    �=v?) ����>2&b՞
�u?*QH�x?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        gN�,�       �E��	��a���A�*�
w
discriminator_loss*a	    ;�o?    ;�o?      �?!    ;�o?) �ܟy��>2�N�W�m?;8�clp?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	9!�a���A�*�
w
discriminator_loss*a	   @��m?   @��m?      �?!   @��m?)��=}��>2ߤ�(g%k?�N�W�m?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        l ��       �E��	q]�a���A�*�
w
discriminator_loss*a	   @�^}?   @�^}?      �?!   @�^}?)�����
?2���T}?>	� �?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Q�-p�       �E��	w��a���A�*�
w
discriminator_loss*a	   �q?   �q?      �?!   �q?) Qo�q4�>2;8�clp?uWy��r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        b0p��       �E��	�4�a���A�*�
w
discriminator_loss*a	   @V?   @V?      �?!   @V?)�x�u�%?2���T}?>	� �?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���p�       �E��	_�Cb���A�*�
w
discriminator_loss*a	   ��as?   ��as?      �?!   ��as?) $��yz�>2uWy��r?hyO�s?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        d}C0�       �E��	��Kb���A�*�
w
discriminator_loss*a	   `�a?   `�a?      �?!   `�a?)@&X�R3�>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]Ō(�       �E��	�mSb���A�*�
w
discriminator_loss*a	   �y	k?   �y	k?      �?!   �y	k?) R%����>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��?�       �E��	��Zb���A�*�
w
discriminator_loss*a	    mf?    mf?      �?!    mf?)@����n�>25Ucv0ed?Tw��Nof?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��4�       �E��	��ab���A�*�
w
discriminator_loss*a	   �wuq?   �wuq?      �?!   �wuq?) �,W��>2;8�clp?uWy��r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        +�$ �       �E��	aib���A�*�
w
discriminator_loss*a	   ���p?   ���p?      �?!   ���p?)@�}���>2;8�clp?uWy��r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ջ~��       �E��	�qb���A�*�
w
discriminator_loss*a	   ���q?   ���q?      �?!   ���q?)@�,b��>2;8�clp?uWy��r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �.ݠ�       �E��	]�{b���A�*�
w
discriminator_loss*a	   �sY?   �sY?      �?!   �sY?) b�z=�>2��bB�SY?�m9�H�[?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        :W�x�       �E��	�7�b���A�*�
w
discriminator_loss*a	   @/�w?   @/�w?      �?!   @/�w?)�Ĺ��r?2&b՞
�u?*QH�x?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��Ct�       �E��	b��b���A�*�
w
discriminator_loss*a	   @ji?   @ji?      �?!   @ji?)��&�0/�>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���)�       �E��	��b���A�*�
w
discriminator_loss*a	   @�!r?   @�!r?      �?!   @�!r?) �(*��>2uWy��r?hyO�s?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ­�	�       �E��	ƅ�b���A�*�
w
discriminator_loss*a	    U�u?    U�u?      �?!    U�u?) ��uBC�>2hyO�s?&b՞
�u?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �WU�       �E��	��b���A�*�
w
discriminator_loss*a	   �lV?   �lV?      �?!   �lV?) Y@�l�>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        S�r�       �E��	��b���A�*�
w
discriminator_loss*a	   ���c?   ���c?      �?!   ���c?) ��1���>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �h�q�       �E��	ҳc���A�*�
w
discriminator_loss*a	   �vk?   �vk?      �?!   �vk?)�Đ���>2ߤ�(g%k?�N�W�m?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��       �E��	�oc���A�*�
w
discriminator_loss*a	   �Y�g?   �Y�g?      �?!   �Y�g?)��'q���>2Tw��Nof?P}���h?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        j{��       �E��	�vhc���A�*�
w
discriminator_loss*a	    4�r?    4�r?      �?!    4�r?)  )��/�>2uWy��r?hyO�s?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	g�oc���A�*�
w
discriminator_loss*a	    ͤf?    ͤf?      �?!    ͤf?) Ha���>2Tw��Nof?P}���h?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        J�.��       �E��	6wc���A�*�
w
discriminator_loss*a	   ���h?   ���h?      �?!   ���h?) b��Z�>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        4�]�       �E��	�5~c���A�*�
w
discriminator_loss*a	    �k?    �k?      �?!    �k?) ]dW��>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        t���       �E��	�3�c���A�*�
w
discriminator_loss*a	   ���b?   ���b?      �?!   ���b?)@�����>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        y{[�       �E��	��c���A�*�
w
discriminator_loss*a	   @"�h?   @"�h?      �?!   @"�h?)���'
?�>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �3�,�       �E��	E�c���A�*�
w
discriminator_loss*a	   �zuZ?   �zuZ?      �?!   �zuZ?) �Y���>2��bB�SY?�m9�H�[?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Tgv��       �E��	�0�c���A�*�
w
discriminator_loss*a	   �R/f?   �R/f?      �?!   �R/f?)@D����>25Ucv0ed?Tw��Nof?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        2����       �E��	� d���A�*�
w
discriminator_loss*a	   `�c?   `�c?      �?!   `�c?)@.��7�>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�	d���A�*�
w
discriminator_loss*a	   ��=c?   ��=c?      �?!   ��=c?) ��I+#�>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Ŧ-c�       �E��	5#d���A�*�
w
discriminator_loss*a	   �;ri?   �;ri?      �?!   �;ri?)��'&<�>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �6�       �E��	qd���A�*�
w
discriminator_loss*a	   @�z[?   @�z[?      �?!   @�z[?)�\]�̘�>2��bB�SY?�m9�H�[?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        S��Z�       �E��	v� d���A�*�
w
discriminator_loss*a	    �]X?    �]X?      �?!    �]X?) ��E���>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��$�       �E��	��'d���A�*�
w
discriminator_loss*a	   �Va?   �Va?      �?!   �Va?) ����>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	e�.d���A�*�
w
discriminator_loss*a	    #�b?    #�b?      �?!    #�b?) �LR��>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        N���       �E��	x�6d���A�*�
w
discriminator_loss*a	   ��-^?   ��-^?      �?!   ��-^?)�XǇ�u�>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        H�       �E��	��d���A�*�
w
discriminator_loss*a	    �Da?    �Da?      �?!    �Da?)@�׈f��>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        \�Ʌ�       �E��	���d���A�*�
w
discriminator_loss*a	   �r�[?   �r�[?      �?!   �r�[?) bʁ}��>2��bB�SY?�m9�H�[?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ڠ��       �E��	���d���A�*�
w
discriminator_loss*a	   �{cj?   �{cj?      �?!   �{cj?)���_���>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        9*�A�       �E��	Y��d���A�*�
w
discriminator_loss*a	   `��^?   `��^?      �?!   `��^?) Y���L�>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �2]�       �E��	�C�d���A�*�
w
discriminator_loss*a	   @��\?   @��\?      �?!   @��\?)� t�G�>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        J~�R�       �E��	���d���A�*�
w
discriminator_loss*a	   �*d?   �*d?      �?!   �*d?)@��pt�>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        F���       �E��	���d���A�*�
w
discriminator_loss*a	    �@U?    �@U?      �?!    �@U?)@��l�;�>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �j�^�       �E��	u�d���A�*�
w
discriminator_loss*a	   ��VV?   ��VV?      �?!   ��VV?) ��{0�>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        }E�	�       �E��	'He���A�*�
w
discriminator_loss*a	   @�>Y?   @�>Y?      �?!   @�>Y?)�lf|��>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �5�
�       �E��	rKOe���A�*�
w
discriminator_loss*a	    TG?    TG?      �?!    TG?) b����>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        )It��       �E��	�lVe���A�*�
w
discriminator_loss*a	   �h�W?   �h�W?      �?!   �h�W?) B�S���>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        N��%�       �E��		~]e���A�*�
w
discriminator_loss*a	   @��V?   @��V?      �?!   @��V?)���0�>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��ۿ�       �E��	��de���A�*�
w
discriminator_loss*a	    \T?    \T?      �?!    \T?) @E �>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?         .�       �E��	%�ke���A�*�
w
discriminator_loss*a	   `L�Z?   `L�Z?      �?!   `L�Z?) I�:��>2��bB�SY?�m9�H�[?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ^\;>�       �E��	�se���A�*�
w
discriminator_loss*a	    ��Q?    ��Q?      �?!    ��Q?)@�L���>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �8�       �E��	3ze���A�*�
w
discriminator_loss*a	    c�G?    c�G?      �?!    c�G?) H����>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        K���       �E��	$��e���A�*�
w
discriminator_loss*a	    |zg?    |zg?      �?!    |zg?) �`�9�>2Tw��Nof?P}���h?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        c����       �E��	��e���A�*�
w
discriminator_loss*a	    PQ?    PQ?      �?!    PQ?)@@�w.�>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ;�k�       �E��	m��e���A�*�
w
discriminator_loss*a	   ��_?   ��_?      �?!   ��_?)����E��>2E��{��^?�l�P�`?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        1�B��       �E��	.�f���A�*�
w
discriminator_loss*a	   `&�`?   `&�`?      �?!   `&�`?)@
�G�z�>2E��{��^?�l�P�`?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �hXd�       �E��	��f���A�*�
w
discriminator_loss*a	   ���i?   ���i?      �?!   ���i?) ���>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        c�-<�       �E��	;�f���A�*�
w
discriminator_loss*a	   @j�D?   @j�D?      �?!   @j�D?) ���=t�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        om�       �E��	"�f���A�*�
w
discriminator_loss*a	   `F{U?   `F{U?      �?!   `F{U?)@�9�N׼>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        uu���       �E��	۲f���A�*�
w
discriminator_loss*a	    �J?    �J?      �?!    �J?) D�hy�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        r�Gt�       �E��	�Z�f���A�*�
w
discriminator_loss*a	   �a?   �a?      �?!   �a?)@�Xos�>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �h���       �E��	�f���A�*�
w
discriminator_loss*a	   ���`?   ���`?      �?!   ���`?)@: �w��>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %!I�       �E��	D'�f���A�*�
w
discriminator_loss*a	   �е\?   �е\?      �?!   �е\?) ���5��>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �A)�       �E��	�,�f���A�*�
w
discriminator_loss*a	   ���H?   ���H?      �?!   ���H?) ��E\"�>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        >-���       �E��	�R�f���A�*�
w
discriminator_loss*a	   ���L?   ���L?      �?!   ���L?)�p�j,�>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��L�       �E��	T��f���A�*�
w
discriminator_loss*a	   ��\L?   ��\L?      �?!   ��\L?) �?��#�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��e�       �E��	���f���A�*�
w
discriminator_loss*a	   �v�N?   �v�N?      �?!   �v�N?)���\�Q�>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �@�E�       �E��	��f���A�*�
w
discriminator_loss*a	    �7Y?    �7Y?      �?!    �7Y?) �{��>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        H��       �E��	�Hg���A�*�
w
discriminator_loss*a	   ���H?   ���H?      �?!   ���H?) @�Z�3�>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        i����       �E��	B	Pg���A�*�
w
discriminator_loss*a	   ��;?   ��;?      �?!   ��;?) VF��>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �]���       �E��	�2Xg���A�*�
w
discriminator_loss*a	   �|�Q?   �|�Q?      �?!   �|�Q?) �w�޳>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        @���       �E��	r.`g���A�*�
w
discriminator_loss*a	    ��]?    ��]?      �?!    ��]?)  �����>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ty��       �E��	}hg���A�*�
w
discriminator_loss*a	   `�Q?   `�Q?      �?!   `�Q?)@Q�[�>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���4�       �E��	��og���A�*�
w
discriminator_loss*a	    �1F?    �1F?      �?!    �1F?) @�6�ɞ>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	=xg���A�*�
w
discriminator_loss*a	   �!�Q?   �!�Q?      �?!   �!�Q?)@�C�׳>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Z��b�       �E��	��g���A�*�
w
discriminator_loss*a	   @FM?   @FM?      �?!   @FM?)�8��w�>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �fV�       �E��	�8h���A�*�
w
discriminator_loss*a	   ���R?   ���R?      �?!   ���R?)@\����>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        3�ϱ�       �E��	�h���A�*�
w
discriminator_loss*a	   �z�X?   �z�X?      �?!   �z�X?)���� �>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��$��       �E��	�h���A�*�
w
discriminator_loss*a	    ��\?    ��\?      �?!    ��\?) H����>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        f�1��       �E��	K�h���A�*�
w
discriminator_loss*a	   `a\E?   `a\E?      �?!   `a\E?)@� ����>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	��'h���A�*�
w
discriminator_loss*a	   �ib?   �ib?      �?!   �ib?) �W.���>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        a��v�       �E��	?R/h���A�*�
w
discriminator_loss*a	   ���T?   ���T?      �?!   ���T?)@���غ>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ѱ��       �E��	�6h���A�*�
w
discriminator_loss*a	    ��L?    ��L?      �?!    ��L?) �T��h�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	7h>h���A�*�
w
discriminator_loss*a	   @��<?   @��<?      �?!   @��<?)��2�0�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        .*#�       �E��	F�h���A�*�
w
discriminator_loss*a	    -K;?    -K;?      �?!    -K;?) H/��G�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?         R���       �E��	�V�h���A�*�
w
discriminator_loss*a	   ���D?   ���D?      �?!   ���D?) �'|�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��       �E��	�R�h���A�*�
w
discriminator_loss*a	   ���S?   ���S?      �?!   ���S?)@��K��>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        &/��       �E��	&�h���A�*�
w
discriminator_loss*a	   ��/2?   ��/2?      �?!   ��/2?) ��3�t>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %���       �E��	D	�h���A�*�
w
discriminator_loss*a	   ��VG?   ��VG?      �?!   ��VG?) 2�wE�>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        {��*�       �E��	�
�h���A�*�
w
discriminator_loss*a	   ��iN?   ��iN?      �?!   ��iN?)�l��8�>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �=��       �E��	z�h���A�*�
w
discriminator_loss*a	   @Y3D?   @Y3D?      �?!   @Y3D?) �����>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ߡ�X�       �E��	�$i���A�*�
w
discriminator_loss*a	    �hX?    �hX?      �?!    �hX?) �m���>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ?����       �E��	���i���A�*�
w
discriminator_loss*a	   @<~\?   @<~\?      �?!   @<~\?)�p�j�^�>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        1����       �E��	��i���A�*�
w
discriminator_loss*a	    �S?    �S?      �?!    �S?)@`p���>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        t���       �E��	zA�i���A�*�
w
discriminator_loss*a	   `J>?   `J>?      �?!   `J>?) ������>2d�\D�X=?���#@?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	wB�i���A�*�
w
discriminator_loss*a	   ��T?   ��T?      �?!   ��T?) 9iO��>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        a_�S�       �E��	Pp�i���A�*�
w
discriminator_loss*a	   �U�B?   �U�B?      �?!   �U�B?) �kΏ��>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        n�T��       �E��	�u�i���A�*�
w
discriminator_loss*a	    �WD?    �WD?      �?!    �WD?)@Dv �ݙ>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ں��       �E��	���i���A�*�
w
discriminator_loss*a	   �L?   �L?      �?!   �L?) 2��¦�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	���i���A�*�
w
discriminator_loss*a	   ��G?   ��G?      �?!   ��G?) ����>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	"�cj���A�*�
w
discriminator_loss*a	   ���G?   ���G?      �?!   ���G?)��6���>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        o_���       �E��	�jj���A�*�
w
discriminator_loss*a	    JQ?    JQ?      �?!    JQ?) @�j.�>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��v�       �E��	��qj���A�*�
w
discriminator_loss*a	   ���;?   ���;?      �?!   ���;?) �^$1W�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        .Π�       �E��	1yj���A�*�
w
discriminator_loss*a	   �i�E?   �i�E?      �?!   �i�E?)@�t���>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        㳥D�       �E��	�j���A�*�
w
discriminator_loss*a	   ��GM?   ��GM?      �?!   ��GM?) �@�˪>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �v�Z�       �E��	<�j���A�*�
w
discriminator_loss*a	   @q�V?   @q�V?      �?!   @q�V?)�̴���>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        y�fw�       �E��	n3�j���A�*�
w
discriminator_loss*a	   ��\?   ��\?      �?!   ��\?)��0T���>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���:�       �E��	�'�j���A�*�
w
discriminator_loss*a	   �vD?   �vD?      �?!   �vD?)@~˳�D�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Nm��       �E��	Y�5k���A�*�
w
discriminator_loss*a	   @�@?   @�@?      �?!   @�@?) 9�M���>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �D��       �E��	��<k���A�*�
w
discriminator_loss*a	    �$@?    �$@?      �?!    �$@?)  @�J�>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        I'?��       �E��	WDk���A�*�
w
discriminator_loss*a	   �Ta?   �Ta?      �?!   �Ta?) D���A�>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �X���       �E��	~Kk���A�*�
w
discriminator_loss*a	   `O�J?   `O�J?      �?!   `O�J?) �R��/�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �=���       �E��	� Rk���A�*�
w
discriminator_loss*a	    �V?    �V?      �?!    �V?) �h��>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �B��       �E��	��Xk���A�*�
w
discriminator_loss*a	   @�+?   @�+?      �?!   @�+?)�����f>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �В��       �E��	h�_k���A�*�
w
discriminator_loss*a	   @	�D?   @	�D?      �?!   @	�D?) Ym�,[�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �v�G�       �E��	��fk���A�*�
w
discriminator_loss*a	   �7+<?   �7+<?      �?!   �7+<?) �޵�ˈ>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ʩK��       �E��	'�l���A�*�
w
discriminator_loss*a	   �еH?   �еH?      �?!   �еH?)���&��>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        &5^x�       �E��	W�l���A�*�
w
discriminator_loss*a	    �SF?    �SF?      �?!    �SF?)@�%�r'�>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�
l���A�*�
w
discriminator_loss*a	   �tC!?   �tC!?      �?!   �tC!?) D �s�R>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        b��_�       M�	�q3l���A*�
w
discriminator_loss*a	    ��8?    ��8?      �?!    ��8?)  mY{�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �}���       ���g	S�:l���A*�
w
discriminator_loss*a	    �K0?    �K0?      �?!    �K0?) /�p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �w��       ���g	 !Bl���A
*�
w
discriminator_loss*a	   @��P?   @��P?      �?!   @��P?) I�L��>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��s�       ���g	VIl���A*�
w
discriminator_loss*a	   ���R?   ���R?      �?!   ���R?) �7�~�>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �w�j�       ���g	yJPl���A*�
w
discriminator_loss*a	   ���I?   ���I?      �?!   ���I?)�N�
�>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        -���       ���g	Ш m���A*�
w
discriminator_loss*a	    �zP?    �zP?      �?!    �zP?)  ��w��>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        .�w��       ���g	r�m���A*�
w
discriminator_loss*a	    ��I?    ��I?      �?!    ��I?) �0��>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       ���g	�vm���A#*�
w
discriminator_loss*a	   ���8?   ���8?      �?!   ���8?) r��q�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �J�       ���g	�mm���A(*�
w
discriminator_loss*a	   ���8?   ���8?      �?!   ���8?) 6�+k�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        X1�7�       ���g	�`m���A-*�
w
discriminator_loss*a	   @iS.?   @iS.?      �?!   @iS.?)�,V�>�l>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �>��       ���g	�u%m���A2*�
w
discriminator_loss*a	   ���9?   ���9?      �?!   ���9?) 9�ߟ�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �T�N�       ���g	�p,m���A7*�
w
discriminator_loss*a	    ��??    ��??      �?!    ��??)  /A��>2d�\D�X=?���#@?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]���       ���g	0n3m���A<*�
w
discriminator_loss*a	   `��R?   `��R?      �?!   `��R?)@�㶵>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        c^o{�       ���g	 ��m���AA*�
w
discriminator_loss*a	    ��4?    ��4?      �?!    ��4?) @��"{>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        s�d3�       ���g	�m���AF*�
w
discriminator_loss*a	    �$@?    �$@?      �?!    �$@?) �?�I�>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��£�       ���g	���m���AK*�
w
discriminator_loss*a	   `��2?   `��2?      �?!   `��2?)@V�J�v>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ;Z���       ���g	��m���AP*�
w
discriminator_loss*a	   @,;?   @,;?      �?!   @,;?)�0M7φ>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       ���g	xIn���AU*�
w
discriminator_loss*a	   ���L?   ���L?      �?!   ���L?) r���>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        C��!�       ���g	�jn���AZ*�
w
discriminator_loss*a	   ��_R?   ��_R?      �?!   ��_R?)@�0�>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Po��       ���g	�qn���A_*�
w
discriminator_loss*a	    v�@?    v�@?      �?!    v�@?)@X���>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ?�u�       ���g	8tn���Ad*�
w
discriminator_loss*a	   ��T6?   ��T6?      �?!   ��T6?) D�%�*>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �4���       ���g	��n���Ai*�
w
discriminator_loss*a	   @F�9?   @F�9?      �?!   @F�9?)�8��ׄ>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �{U�       ���g	]��n���An*�
w
discriminator_loss*a	   ��C?   ��C?      �?!   ��C?) ��[���>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �a\�       ���g	/n�n���As*�
w
discriminator_loss*a	   �v�A?   �v�A?      �?!   �v�A?) ��R_��>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���>�       ���g	�^�n���Ax*�
w
discriminator_loss*a	   ���+?   ���+?      �?!   ���+?) �~��g>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �;a�       ���g	�c�n���A}*�
w
discriminator_loss*a	   ��YB?   ��YB?      �?!   ��YB?)@
�8"�>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        0B �       �E��	<d�n���A�*�
w
discriminator_loss*a	   �iEC?   �iEC?      �?!   �iEC?)@��6�>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        4�gk�       �E��	�go���A�*�
w
discriminator_loss*a	   ��n5?   ��n5?      �?!   ��n5?) D�ɗ�|>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        S���       �E��	��o���A�*�
w
discriminator_loss*a	    K�9?    K�9?      �?!    K�9?) ȟXʀ�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        f�hz�       �E��	���o���A�*�
w
discriminator_loss*a	   ���A?   ���A?      �?!   ���A?) �&����>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	�Y�o���A�*�
w
discriminator_loss*a	    �1?    �1?      �?!    �1?)  �0aFr>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        i�"��       �E��	�o���A�*�
w
discriminator_loss*a	   �-?   �-?      �?!   �-?) �mj>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��:��       �E��	�o���A�*�
w
discriminator_loss*a	   �)8P?   �)8P?      �?!   �)8P?)@���q�>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ?�C9�       �E��	,�o���A�*�
w
discriminator_loss*a	   @1]7?   @1]7?      �?!   @1]7?)�̏2�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        QX�R�       �E��	��o���A�*�
w
discriminator_loss*a	    �!F?    �!F?      �?!    �!F?) �>�K��>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�	�o���A�*�
w
discriminator_loss*a	   �h0B?   �h0B?      �?!   �h0B?) �-'~��>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �v�=�       �E��	W�o���A�*�
w
discriminator_loss*a	   `	�:?   `	�:?      �?!   `	�:?) �:8~�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �p��       �E��	���p���A�*�
w
discriminator_loss*a	   �if2?   �if2?      �?!   �if2?)@����(u>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���D�       �E��	�r�p���A�*�
w
discriminator_loss*a	   �e3?   �e3?      �?!   �e3?) ��d�v>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        d�G�       �E��	��p���A�*�
w
discriminator_loss*a	    ��C?    ��C?      �?!    ��C?)@��Δ�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	���p���A�*�
w
discriminator_loss*a	   @X"?   @X"?      �?!   @X"?) A���U>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        "���       �E��	���p���A�*�
w
discriminator_loss*a	   ��{4?   ��{4?      �?!   ��{4?) ���9z>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        BI�       �E��	���p���A�*�
w
discriminator_loss*a	   ���(?   ���(?      �?!   ���(?) �!���b>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        YDf�       �E��	'��p���A�*�
w
discriminator_loss*a	   �F8?   �F8?      �?!   �F8?) ���j�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        X�L�       �E��	3��p���A�*�
w
discriminator_loss*a	    NF?    NF?      �?!    NF?) @|��V�>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        IOr��       �E��	���q���A�*�
w
discriminator_loss*a	   ��D?   ��D?      �?!   ��D?)@�9-�F�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �&R�       �E��	��q���A�*�
w
discriminator_loss*a	   ��F?   ��F?      �?!   ��F?) �-Vq�>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	N �q���A�*�
w
discriminator_loss*a	    Y	F?    Y	F?      �?!    Y	F?)@t4��Y�>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	��q���A�*�
w
discriminator_loss*a	   @	"K?   @	"K?      �?!   @	"K?)���ϓ�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        =_���       �E��	i,�q���A�*�
w
discriminator_loss*a	   ��O)?   ��O)?      �?!   ��O)?) ���d>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        L�3�       �E��	z�q���A�*�
w
discriminator_loss*a	   @�B?   @�B?      �?!   @�B?) ���!��>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	�q���A�*�
w
discriminator_loss*a	   ��A0?   ��A0?      �?!   ��A0?) ��+�p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        9��       �E��	&�q���A�*�
w
discriminator_loss*a	   �= 0?   �= 0?      �?!   �= 0?)@Z�@{ p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #j��       �E��	�S�r���A�*�
w
discriminator_loss*a	   �=�:?   �=�:?      �?!   �=�:?)�(C��/�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�m�r���A�*�
w
discriminator_loss*a	   @X�@?   @X�@?      �?!   @X�@?) ��[��>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	���r���A�*�
w
discriminator_loss*a	   �J�1?   �J�1?      �?!   �J�1?)@�4�s>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        9�2�       �E��	��r���A�*�
w
discriminator_loss*a	   ��H?   ��H?      �?!   ��H?) �jڢ>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	���r���A�*�
w
discriminator_loss*a	   `�4C?   `�4C?      �?!   `�4C?)@2��f�>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �YQ�       �E��	���r���A�*�
w
discriminator_loss*a	   �u7?   �u7?      �?!   �u7?)�H�vʩ�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	��s���A�*�
w
discriminator_loss*a	   ���<?   ���<?      �?!   ���<?)�lk_Hŉ>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	&�
s���A�*�
w
discriminator_loss*a	    3�E?    3�E?      �?!    3�E?) ������>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        e�x%�       �E��	��s���A�*�
w
discriminator_loss*a	   �<�<?   �<�<?      �?!   �<�<?) ��>��>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        {��       �E��	���s���A�*�
w
discriminator_loss*a	   ��<?   ��<?      �?!   ��<?) �ϲ-��>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �6(�       �E��	���s���A�*�
w
discriminator_loss*a	   ���!?   ���!?      �?!   ���!?)@\`e�CS>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �_Ծ�       �E��	��s���A�*�
w
discriminator_loss*a	    �#?    �#?      �?!    �#?) �4P��W>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        RVG�       �E��	��t���A�*�
w
discriminator_loss*a	   ��3?   ��3?      �?!   ��3?) A�oYx>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	��t���A�*�
w
discriminator_loss*a	   �Ul#?   �Ul#?      �?!   �Ul#?) �k/)�W>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��̰�       �E��	��t���A�*�
w
discriminator_loss*a	   ��!?   ��!?      �?!   ��!?)@�-6�@R>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Q�>��       �E��	��t���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) r.�y%H>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #��y�       �E��	�u���A�*�
w
discriminator_loss*a	   @�G1?   @�G1?      �?!   @�G1?) ѹ��r>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	z�u���A�*�
w
discriminator_loss*a	    �@?    �@?      �?!    �@?)  j��>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �A�       �E��	Oju���A�*�
w
discriminator_loss*a	   �߫"?   �߫"?      �?!   �߫"?)@����U>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �,,e�       �E��	uou���A�*�
w
discriminator_loss*a	   �6?   �6?      �?!   �6?) 1`�w~>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        b����       �E��	}\&u���A�*�
w
discriminator_loss*a	   `��+?   `��+?      �?!   `��+?) ���@h>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7�h�       �E��	�]-u���A�*�
w
discriminator_loss*a	   ��V5?   ��V5?      �?!   ��V5?) y���t|>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �3�       �E��	Z4u���A�*�
w
discriminator_loss*a	    =�.?    =�.?      �?!    =�.?) �
�6m>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7E.��       �E��	�Y;u���A�*�
w
discriminator_loss*a	    ^K0?    ^K0?      �?!    ^K0?) @h�p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        -E!9�       �E��	AY%v���A�*�
w
discriminator_loss*a	   �x4#?   �x4#?      �?!   �x4#?) ��BJW>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %up��       �E��	I6-v���A�*�
w
discriminator_loss*a	   @2�+?   @2�+?      �?!   @2�+?)��޳�h>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �B�`�       �E��	)�4v���A�*�
w
discriminator_loss*a	   �)UC?   �)UC?      �?!   �)UC?) �t\�>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #j���       �E��	��;v���A�*�
w
discriminator_loss*a	    �Q9?    �Q9?      �?!    �Q9?) �sh�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        H�%��       �E��	uCv���A�*�
w
discriminator_loss*a	   �� %?   �� %?      �?!   �� %?)@٭_�[>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        K�j��       �E��	E,Jv���A�*�
w
discriminator_loss*a	   �c�?   �c�?      �?!   �c�?) ���1>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        +��L�       �E��	�Qv���A�*�
w
discriminator_loss*a	   ��"?   ��"?      �?!   ��"?)@F��V>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        |��p�       �E��	� Xv���A�*�
w
discriminator_loss*a	   ��Q/?   ��Q/?      �?!   ��Q/?)��D|��n>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�Hw���A�*�
w
discriminator_loss*a	    �-?    �-?      �?!    �-?) *��pj>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        A@�p�       �E��	��Pw���A�*�
w
discriminator_loss*a	   ��X)?   ��X)?      �?!   ��X)?)��Mjd>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �8�2�       �E��	�Ww���A�*�
w
discriminator_loss*a	   �a�I?   �a�I?      �?!   �a�I?) Օ�h�>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	{�^w���A�*�
w
discriminator_loss*a	    �b1?    �b1?      �?!    �b1?) � ��r>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        2$���       �E��	d�ew���A�*�
w
discriminator_loss*a	   ���3?   ���3?      �?!   ���3?)@f��w>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �G0�       �E��	��lw���A�*�
w
discriminator_loss*a	   �m�D?   �m�D?      �?!   �m�D?) d��F�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���4�       �E��	�sw���A�*�
w
discriminator_loss*a	    �u9?    �u9?      �?!    �u9?) ��W�A�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        /5l��       �E��	G�zw���A�*�
w
discriminator_loss*a	   `:_$?   `:_$?      �?!   `:_$?)@��H�Y>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        _%vq�       �E��	v`tx���A�*�
w
discriminator_loss*a	   ���5?   ���5?      �?!   ���5?) A?��}>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��L��       �E��	�|x���A�*�
w
discriminator_loss*a	   @��0?   @��0?      �?!   @��0?) �W��q>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        3����       �E��	!x���A�*�
w
discriminator_loss*a	   �	�;?   �	�;?      �?!   �	�;?) �P�=�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        E�u��       �E��	^Պx���A�*�
w
discriminator_loss*a	   � �6?   � �6?      �?!   � �6?) �5u�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        G�j�       �E��	Xɑx���A�*�
w
discriminator_loss*a	   ��� ?   ��� ?      �?!   ��� ?) �ѯ��Q>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ޗ�       �E��	gɘx���A�*�
w
discriminator_loss*a	    �Z0?    �Z0?      �?!    �Z0?)@ta@3�p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	˟x���A�*�
w
discriminator_loss*a	   @|1?   @|1?      �?!   @|1?) ��I�r>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ك���       �E��	�ͦx���A�*�
w
discriminator_loss*a	   ��_?   ��_?      �?!   ��_?) �Ž��9>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �P��       �E��	�i�y���A�*�
w
discriminator_loss*a	    N�1?    N�1?      �?!    N�1?) @|��
t>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �*P$�       �E��	'7�y���A�*�
w
discriminator_loss*a	    n ?    n ?      �?!    n ?) |�3�P>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Q+��       �E��	H��y���A�*�
w
discriminator_loss*a	   �\.+?   �\.+?      �?!   �\.+?) ��g>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        k
���       �E��	圷y���A�*�
w
discriminator_loss*a	   `ױ%?   `ױ%?      �?!   `ױ%?)@&�j]>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        v8�A�       �E��	��y���A�*�
w
discriminator_loss*a	   @�8?   @�8?      �?!   @�8?)���R�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        A��       �E��	G��y���A�*�
w
discriminator_loss*a	   ���+?   ���+?      �?!   ���+?)��P5�g>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        dP���       �E��	���y���A�*�
w
discriminator_loss*a	   ��0?   ��0?      �?!   ��0?)@� k��q>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��t:�       �E��	:��y���A�*�
w
discriminator_loss*a	    ��#?    ��#?      �?!    ��#?)@�t��xX>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        S�=0�       �E��	l�z���A�*�
w
discriminator_loss*a	   @�Z%?   @�Z%?      �?!   @�Z%?) AX�\>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��ϗ�       �E��	C|�z���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) ɣrj4>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        l&o�       �E��	A+�z���A�*�
w
discriminator_loss*a	    �2?    �2?      �?!    �2?) @^�Тu>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �M;��       �E��	GI {���A�*�
w
discriminator_loss*a	   �(&?   �(&?      �?!   �(&?) Ā�m�^>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	{�{���A�*�
w
discriminator_loss*a	   ��+?   ��+?      �?!   ��+?) �<�1h>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �� �       �E��	L�{���A�*�
w
discriminator_loss*a	    �@7?    �@7?      �?!    �@7?) 4ԸH�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��g>�       �E��	3�{���A�*�
w
discriminator_loss*a	   ��F/?   ��F/?      �?!   ��F/?)����W�n>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Z$��       �E��	�{���A�*�
w
discriminator_loss*a	   ��2?   ��2?      �?!   ��2?)@T�Pnt>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        p�8�       �E��	��?|���A�*�
w
discriminator_loss*a	   �7?   �7?      �?!   �7?) ��\���>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �d��       �E��	�XG|���A�*�
w
discriminator_loss*a	   �Be?   �Be?      �?!   �Be?) d�xV&5>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �s}��       �E��	/�O|���A�*�
w
discriminator_loss*a	   @��(?   @��(?      �?!   @��(?)�P�xWLc>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �t��       �E��	ъV|���A�*�
w
discriminator_loss*a	   �m�?   �m�?      �?!   �m�?) �޽|M>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        h/��       �E��	K�]|���A�*�
w
discriminator_loss*a	   ൄ0?   ൄ0?      �?!   ൄ0?)@h��q>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        :���       �E��	�_e|���A�*�
w
discriminator_loss*a	   @jQ+?   @jQ+?      �?!   @jQ+?)��Do2Rg>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        k%)��       �E��	E�l|���A�*�
w
discriminator_loss*a	   @c#"?   @c#"?      �?!   @c#"?) ����T>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �:��       �E��	�s|���A�*�
w
discriminator_loss*a	    W';?    W';?      �?!    W';?) 6Kc�
�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        K~K��       �E��	��}���A�*�
w
discriminator_loss*a	   ���)?   ���)?      �?!   ���)?) ���f�d>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��y��       �E��	E�}���A�*�
w
discriminator_loss*a	   �}�?   �}�?      �?!   �}�?) $���O>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �"��       �E��	�Ū}���A�*�
w
discriminator_loss*a	   ���D?   ���D?      �?!   ���D?) 	��ݚ>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �/�       �E��	~N�}���A�*�
w
discriminator_loss*a	    @�(?    @�(?      �?!    @�(?) �����b>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        w/��       �E��	�}���A�*�
w
discriminator_loss*a	   @{�3?   @{�3?      �?!   @{�3?) i�Wz�x>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        m&<�       �E��	� �}���A�*�
w
discriminator_loss*a	   �,.?   �,.?      �?!   �,.?)��^+fMl>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	B��}���A�*�
w
discriminator_loss*a	   �23?   �23?      �?!   �23?)@���w>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �m�       �E��	���}���A�*�
w
discriminator_loss*a	   �~Q?   �~Q?      �?!   �~Q?) \=�E>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �h,�       �E��	���A�*�
w
discriminator_loss*a	   �p�?   �p�?      �?!   �p�?) ����B>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �F�       �E��	yR���A�*�
w
discriminator_loss*a	    ��#?    ��#?      �?!    ��#?)  ���X>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        g�a�       �E��	�:!���A�*�
w
discriminator_loss*a	   ��09?   ��09?      �?!   ��09?) $~��ԃ>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �_L�       �E��	a�(���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) �Q��H>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��Y��       �E��	��0���A�*�
w
discriminator_loss*a	   ���@?   ���@?      �?!   ���@?) y׎�O�>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ,���       �E��	X�8���A�*�
w
discriminator_loss*a	    D'?    D'?      �?!    D'?) ��s��`>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �,�,�       �E��	�n@���A�*�
w
discriminator_loss*a	   ���-?   ���-?      �?!   ���-?) r�u��k>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �@���       �E��	�?H���A�*�
w
discriminator_loss*a	   `EM!?   `EM!?      �?!   `EM!?)@�蘨�R>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �i�~�       �E��	������A�*�
w
discriminator_loss*a	    wX&?    wX&?      �?!    wX&?)@�ֹ05_>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��	��       �E��	ᑀ���A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)���q�M>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	w������A�*�
w
discriminator_loss*a	    gS'?    gS'?      �?!    gS'?) VB�� a>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �� u�       �E��	�������A�*�
w
discriminator_loss*a	    )L'?    )L'?      �?!    )L'?) ڌa0�`>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �-���       �E��	qé����A�*�
w
discriminator_loss*a	   �N�5?   �N�5?      �?!   �N�5?)@����}>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��Ќ�       �E��	�t�����A�*�
w
discriminator_loss*a	   @��+?   @��+?      �?!   @��+?)���ᶤg>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �C���       �E��	�{�����A�*�
w
discriminator_loss*a	   �U*?   �U*?      �?!   �U*?)���N�H>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7HX�       �E��	g������A�*�
w
discriminator_loss*a	    [�#?    [�#?      �?!    [�#?) ����kX>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��tO�       �E��	b�"����A�*�
w
discriminator_loss*a	   ��*?   ��*?      �?!   ��*?) B��Ce>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        [��       �E��	��*����A�*�
w
discriminator_loss*a	   @�i?   @�i?      �?!   @�i?) I�@�e?>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �铷�       �E��	KD2����A�*�
w
discriminator_loss*a	   ��`?   ��`?      �?!   ��`?)�d��J>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �mA��       �E��	�`9����A�*�
w
discriminator_loss*a	   @4�?   @4�?      �?!   @4�?)�P���)O>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        X��       �E��	du@����A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)����r�A>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #�.��       �E��	H�G����A�*�
w
discriminator_loss*a	    Ze(?    Ze(?      �?!    Ze(?)  H�b>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        $4���       �E��	�	O����A�*�
w
discriminator_loss*a	   �@�"?   �@�"?      �?!   �@�"?)@����U>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        q��       �E��	�V����A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) @���6>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        6I��       �E��	������A�*�
w
discriminator_loss*a	   ��W/?   ��W/?      �?!   ��W/?)����n>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���S�       �E��	�?�����A�*�
w
discriminator_loss*a	   `K[?   `K[?      �?!   `K[?) �c�.>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        D�/�       �E��	̼����A�*�
w
discriminator_loss*a	    �~?    �~?      �?!    �~?)  ȍ�OD>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�~ă���A�*�
w
discriminator_loss*a	   `�j?   `�j?      �?!   `�j?)@��4=:>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �pW,�       �E��	�G̃���A�*�
w
discriminator_loss*a	   `��1?   `��1?      �?!   `��1?)@N�l��s>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        o����       �E��	CWԃ���A�*�
w
discriminator_loss*a	   �Q�?   �Q�?      �?!   �Q�?) 5�"E7H>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �߅)�       �E��	��ۃ���A�*�
w
discriminator_loss*a	   ��r%?   ��r%?      �?!   ��r%?)@��m�\>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        =����       �E��	�(ヨ��A�*�
w
discriminator_loss*a	    ��*?    ��*?      �?!    ��*?) �WK�f>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        \�p��       �E��	h�=����A�*�
w
discriminator_loss*a	    �J3?    �J3?      �?!    �J3?)@hPNcBw>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        L#X�       �E��	��F����A�*�
w
discriminator_loss*a	   �0?   �0?      �?!   �0?)�D��v�@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �]Gk�       �E��	OO����A�*�
w
discriminator_loss*a	   ��69?   ��69?      �?!   ��69?) ���݃>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �EX�       �E��	��V����A�*�
w
discriminator_loss*a	   ��'?   ��'?      �?!   ��'?)���7��`>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        r!��       �E��	_����A�*�
w
discriminator_loss*a	   @�!?   @�!?      �?!   @�!?) ы�q�S>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        &^�u�       �E��	VCf����A�*�
w
discriminator_loss*a	   @�D?   @�D?      �?!   @�D?)�� &�N>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �hB��       �E��	��m����A�*�
w
discriminator_loss*a	   �r�"?   �r�"?      �?!   �r�"?) d���U>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ߒS�       M�	I������A*�
w
discriminator_loss*a	   @w('?   @w('?      �?!   @w('?)�dܗ^�`>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �7��       ���g	2�ц���A*�
w
discriminator_loss*a	   `C�?   `C�?      �?!   `C�?) ���UO>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���e�       ���g	\�ن���A
*�
w
discriminator_loss*a	   @Ʉ?   @Ʉ?      �?!   @Ʉ?)��A��:K>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���W�       ���g	`T↨��A*�
w
discriminator_loss*a	   �d�?   �d�?      �?!   �d�?) ���|�)>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �6�       ���g	�Hꆨ��A*�
w
discriminator_loss*a	   @J?   @J?      �?!   @J?)�H��9*E>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��8�       ���g	�����A*�
w
discriminator_loss*a	   �?�?   �?�?      �?!   �?�?)� �@��O>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �4���       ���g	�M�����A*�
w
discriminator_loss*a	   ���$?   ���$?      �?!   ���$?) ���[>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ф�       ���g	r?����A#*�
w
discriminator_loss*a	   ��3?   ��3?      �?!   ��3?) �p��v>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        $�j�       ���g	��	����A(*�
w
discriminator_loss*a	   �ݧ0?   �ݧ0?      �?!   �ݧ0?) d�+�Vq>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���0�       ���g	}�o����A-*�
w
discriminator_loss*a	   ���%?   ���%?      �?!   ���%?) ��MU]>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        m�       ���g	,�w����A2*�
w
discriminator_loss*a	   ��*?   ��*?      �?!   ��*?) ib�x/f>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        r��0�       ���g	�~����A7*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) ��$"F>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        'E^�       ���g	|�����A<*�
w
discriminator_loss*a	   @��%?   @��%?      �?!   @��%?) ����]>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        pE�       ���g	Vp�����AA*�
w
discriminator_loss*a	   ��t0?   ��t0?      �?!   ��t0?)@ΐ���p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        d4���       ���g	+������AF*�
w
discriminator_loss*a	    h"&?    h"&?      �?!    h"&?)  ���^>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��Z��       ���g	짟����AK*�
w
discriminator_loss*a	   �މ?   �މ?      �?!   �މ?) $fn�>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ks�       ���g	"m�����AP*�
w
discriminator_loss*a	   �bo:?   �bo:?      �?!   �bo:?) 2�ăօ>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �xգ�       ���g	ŵ����AU*�
w
discriminator_loss*a	    ��	?    ��	?      �?!    ��	?) !f��$>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        _*���       ���g	�i����AZ*�
w
discriminator_loss*a	   �Rs*?   �Rs*?      �?!   �Rs*?) ��H�e>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       ���g	<�����A_*�
w
discriminator_loss*a	    H�?    H�?      �?!    H�?)  �`�I>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        !d���       ���g	�����Ad*�
w
discriminator_loss*a	   �Q�?   �Q�?      �?!   �Q�?)������->2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �I��       ���g	,�$����Ai*�
w
discriminator_loss*a	   ���<?   ���<?      �?!   ���<?) ��k;�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       ���g	��+����An*�
w
discriminator_loss*a	   ��'?   ��'?      �?!   ��'?)��#��@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �A�}�       ���g	f�3����As*�
w
discriminator_loss*a	   ��#?   ��#?      �?!   ��#?)@R�B^X>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �C���       ���g	ٹ:����Ax*�
w
discriminator_loss*a	   �	�?   �	�?      �?!   �	�?) � ��D>2�