       �K"	   ֩��Abrain.Event:2���     �;�	Y/'֩��A"��
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
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/sub*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�*
T0
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
<Generator/first/Generator/firstfully_connected/kernel/AssignAssign5Generator/first/Generator/firstfully_connected/kernelPGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
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
5Generator/first/Generator/firstfully_connected/MatMulMatMulGenerator/noise:Generator/first/Generator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
6Generator/first/Generator/firstfully_connected/BiasAddBiasAdd5Generator/first/Generator/firstfully_connected/MatMul8Generator/first/Generator/firstfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
t
/Generator/first/Generator/firstleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *��L>*
dtype0
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
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
�
`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformXGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shape*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
seed2*
dtype0* 
_output_shapes
:
��*

seed*
T0
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
>Generator/second/Generator/secondfully_connected/kernel/AssignAssign7Generator/second/Generator/secondfully_connected/kernelRGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
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
<Generator/second/Generator/secondfully_connected/bias/AssignAssign5Generator/second/Generator/secondfully_connected/biasGGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
:Generator/second/Generator/secondfully_connected/bias/readIdentity5Generator/second/Generator/secondfully_connected/bias*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:�
�
7Generator/second/Generator/secondfully_connected/MatMulMatMul)Generator/first/Generator/firstleaky_relu<Generator/second/Generator/secondfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
8Generator/second/Generator/secondfully_connected/BiasAddBiasAdd7Generator/second/Generator/secondfully_connected/MatMul:Generator/second/Generator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
HGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/onesConst*
_output_shapes	
:�*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB�*  �?*
dtype0
�
7Generator/second/Generator/secondbatch_normalized/gamma
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma
�
>Generator/second/Generator/secondbatch_normalized/gamma/AssignAssign7Generator/second/Generator/secondbatch_normalized/gammaHGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma
�
<Generator/second/Generator/secondbatch_normalized/gamma/readIdentity7Generator/second/Generator/secondbatch_normalized/gamma*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:�
�
HGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB�*    
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
BGenerator/second/Generator/secondbatch_normalized/moving_mean/readIdentity=Generator/second/Generator/secondbatch_normalized/moving_mean*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
_output_shapes	
:�*
T0
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
?Generator/second/Generator/secondbatch_normalized/batchnorm/mulMulAGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt<Generator/second/Generator/secondbatch_normalized/gamma/read*
_output_shapes	
:�*
T0
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAdd?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2MulBGenerator/second/Generator/secondbatch_normalized/moving_mean/read?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:�
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
1Generator/second/Generator/secondleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
/Generator/second/Generator/secondleaky_relu/mulMul1Generator/second/Generator/secondleaky_relu/alphaAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
+Generator/second/Generator/secondleaky_reluMaximum/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
VGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      
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
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��
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
:Generator/third/Generator/thirdfully_connected/kernel/readIdentity5Generator/third/Generator/thirdfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container 
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
8Generator/third/Generator/thirdfully_connected/bias/readIdentity3Generator/third/Generator/thirdfully_connected/bias*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:�
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
FGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/onesConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
5Generator/third/Generator/thirdbatch_normalized/gamma
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
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
:Generator/third/Generator/thirdbatch_normalized/gamma/readIdentity5Generator/third/Generator/thirdbatch_normalized/gamma*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:�
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
;Generator/third/Generator/thirdbatch_normalized/beta/AssignAssign4Generator/third/Generator/thirdbatch_normalized/betaFGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean
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
VariableV2*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignAssign?Generator/third/Generator/thirdbatch_normalized/moving_variancePGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones*
_output_shapes	
:�*
use_locking(*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
validate_shape(
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
?Generator/third/Generator/thirdbatch_normalized/batchnorm/RsqrtRsqrt=Generator/third/Generator/thirdbatch_normalized/batchnorm/add*
_output_shapes	
:�*
T0
�
=Generator/third/Generator/thirdbatch_normalized/batchnorm/mulMul?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt:Generator/third/Generator/thirdbatch_normalized/gamma/read*
_output_shapes	
:�*
T0
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
VGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      
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
^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
seed2m*
dtype0* 
_output_shapes
:
��*

seed
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
PGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniformAddTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��*
T0
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
:Generator/forth/Generator/forthfully_connected/kernel/readIdentity5Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
�
UGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�*
dtype0
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
EGenerator/forth/Generator/forthfully_connected/bias/Initializer/zerosFillUGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorKGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/Const*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0
�
3Generator/forth/Generator/forthfully_connected/bias
VariableV2*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
VGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorConst*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:�*
dtype0
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
<Generator/forth/Generator/forthbatch_normalized/gamma/AssignAssign5Generator/forth/Generator/forthbatch_normalized/gammaFGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(
�
:Generator/forth/Generator/forthbatch_normalized/gamma/readIdentity5Generator/forth/Generator/forthbatch_normalized/gamma*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:�
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
VariableV2*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:�*
dtype0
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
9Generator/forth/Generator/forthbatch_normalized/beta/readIdentity4Generator/forth/Generator/forthbatch_normalized/beta*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:�*
T0
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
MGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/Const*
_output_shapes	
:�*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*

index_type0
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
`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorConst*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB:�*
dtype0*
_output_shapes
:
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
	container *
shape:�
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
=Generator/forth/Generator/forthbatch_normalized/batchnorm/addAddDGenerator/forth/Generator/forthbatch_normalized/moving_variance/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:�
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/RsqrtRsqrt=Generator/forth/Generator/forthbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:�
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
=Generator/forth/Generator/forthbatch_normalized/batchnorm/subSub9Generator/forth/Generator/forthbatch_normalized/beta/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1Add?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/sub*(
_output_shapes
:����������*
T0
t
/Generator/forth/Generator/forthleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
-Generator/forth/Generator/forthleaky_relu/mulMul/Generator/forth/Generator/forthleaky_relu/alpha?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
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
_output_shapes
:
��*

seed*
T0*)
_class
loc:@Generator/dense/kernel*
seed2�*
dtype0
�
5Generator/dense/kernel/Initializer/random_uniform/subSub5Generator/dense/kernel/Initializer/random_uniform/max5Generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@Generator/dense/kernel*
_output_shapes
: 
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
Generator/dense/kernel/AssignAssignGenerator/dense/kernel1Generator/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
��
�
Generator/dense/kernel/readIdentityGenerator/dense/kernel*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
��
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
Generator/dense/bias/AssignAssignGenerator/dense/bias&Generator/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias
�
Generator/dense/bias/readIdentityGenerator/dense/bias*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:�*
T0
�
Generator/dense/MatMulMatMul)Generator/forth/Generator/forthleaky_reluGenerator/dense/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
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
shape:����������*
dtype0*(
_output_shapes
:����������
�
^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0
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
_output_shapes
:
��*

seed*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
seed2�*
dtype0
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/subSub\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/max\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
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
DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignAssign=Discriminator/first/Discriminator/firstfully_connected/kernelXDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
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
=Discriminator/first/Discriminator/firstfully_connected/MatMulMatMulDiscriminator/realBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
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
`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0
�
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *���*
dtype0
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
hDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shape*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
seed2�*
dtype0* 
_output_shapes
:
��*

seed
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
ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniformAdd^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mul^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��*
T0
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
FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignAssign?Discriminator/second/Discriminator/secondfully_connected/kernelZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:�
�
DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignAssign=Discriminator/second/Discriminator/secondfully_connected/biasODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(
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
@Discriminator/second/Discriminator/secondfully_connected/BiasAddBiasAdd?Discriminator/second/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
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
3Discriminator/second/Discriminator/secondleaky_reluMaximum7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
9Discriminator/out/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@Discriminator/out/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
7Discriminator/out/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv�*
dtype0
�
7Discriminator/out/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv>*
dtype0
�
ADiscriminator/out/kernel/Initializer/random_uniform/RandomUniformRandomUniform9Discriminator/out/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	�*

seed*
T0*+
_class!
loc:@Discriminator/out/kernel*
seed2�
�
7Discriminator/out/kernel/Initializer/random_uniform/subSub7Discriminator/out/kernel/Initializer/random_uniform/max7Discriminator/out/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*+
_class!
loc:@Discriminator/out/kernel
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
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	�
�
Discriminator/out/kernel/AssignAssignDiscriminator/out/kernel3Discriminator/out/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel
�
Discriminator/out/kernel/readIdentityDiscriminator/out/kernel*
_output_shapes
:	�*
T0*+
_class!
loc:@Discriminator/out/kernel
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
Discriminator/out/bias/AssignAssignDiscriminator/out/bias(Discriminator/out/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:
�
Discriminator/out/bias/readIdentityDiscriminator/out/bias*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
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
Discriminator/out/SigmoidSigmoidDiscriminator/out/BiasAdd*'
_output_shapes
:���������*
T0
�
?Discriminator/first_1/Discriminator/firstfully_connected/MatMulMatMulGenerator/TanhBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
@Discriminator/first_1/Discriminator/firstfully_connected/BiasAddBiasAdd?Discriminator/first_1/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
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
ADiscriminator/second_1/Discriminator/secondfully_connected/MatMulMatMul3Discriminator/first_1/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
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
9Discriminator/second_1/Discriminator/secondleaky_relu/mulMul;Discriminator/second_1/Discriminator/secondleaky_relu/alphaBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
5Discriminator/second_1/Discriminator/secondleaky_reluMaximum9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Discriminator/out_1/MatMulMatMul5Discriminator/second_1/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
Discriminator/out_1/BiasAddBiasAddDiscriminator/out_1/MatMulDiscriminator/out/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
u
Discriminator/out_1/SigmoidSigmoidDiscriminator/out_1/BiasAdd*
T0*'
_output_shapes
:���������
W
LogLogDiscriminator/out/Sigmoid*'
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
MeanMeanaddConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
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
sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
Mean_1MeanLog_2Const_1*
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
generator_lossHistogramSummarygenerator_loss/tagMean_1*
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
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
^
gradients/Mean_grad/Shape_1Shapeadd*
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
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Mean_grad/truediv(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sumgradients/Mean_grad/truediv*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*'
_output_shapes
:���������
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
4gradients/Discriminator/out/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out/Sigmoidgradients/Log_grad/mul*'
_output_shapes
:���������*
T0
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
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
_output_shapes
: *
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
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
6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid-gradients/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
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
Bgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Identity0gradients/Discriminator/out/MatMul_grad/MatMul_19^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1
�
6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
�
;gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad7^gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
�
Cgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������
�
Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeShape7Discriminator/second/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
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
Ogradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Xgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Fgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumSumIgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectXgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1SumKgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1Zgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
0gradients/Discriminator/out_1/MatMul_grad/MatMulMatMulCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
2gradients/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	�*
transpose_a(
�
:gradients/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp1^gradients/Discriminator/out_1/MatMul_grad/MatMul3^gradients/Discriminator/out_1/MatMul_grad/MatMul_1
�
Bgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity0gradients/Discriminator/out_1/MatMul_grad/MatMul;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:����������
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
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul9Discriminator/second/Discriminator/secondleaky_relu/alpha[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*c
_classY
WUloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1
�
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
�
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Qgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Zgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumMgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1\gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
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
gradients/AddN_1AddNBgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
N*
_output_shapes
:	�*
T0
�
gradients/AddN_2AddN]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
_output_shapes	
:�*
T0*
data_formatNHWC
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
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
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
Ugradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Wgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul1Discriminator/first/Discriminator/firstleaky_reluhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
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
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Shapeggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
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
Mgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
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
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeReshapeDgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
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
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Zgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulMulYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
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
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1\gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ngradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1V^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
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
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapeigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Ogradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Xgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Fgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumIgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectXgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1T^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
gradients/AddN_5AddNigradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
��
�
gradients/AddN_6AddN[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Ygradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
data_formatNHWC*
_output_shapes	
:�*
T0
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
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
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
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Pgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Wgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
�
_gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
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
egradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentitySgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*f
_class\
ZXloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
ggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityUgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_7AddN]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������*
T0
�
[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
T0*
data_formatNHWC*
_output_shapes	
:�
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
ggradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*h
_class^
\Zloc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul
�
igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
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
VariableV2*
_output_shapes
: *
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape: *
dtype0
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
beta1_power/readIdentitybeta1_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
�
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *w�?
�
beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
�
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
dDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
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
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0
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
KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignAssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(
�
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/readIdentityDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
�
RDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
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
GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignAssign@Discriminator/first/Discriminator/firstfully_connected/bias/AdamRDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    *
dtype0
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
VariableV2*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
�
MDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/AssignAssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/readIdentityFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��
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
GDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/readIdentityBDiscriminator/second/Discriminator/secondfully_connected/bias/Adam*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:�
�
VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB�*    *
dtype0
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
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	�
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
VariableV2*+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name 
�
&Discriminator/out/kernel/Adam_1/AssignAssignDiscriminator/out/kernel/Adam_11Discriminator/out/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	�
�
$Discriminator/out/kernel/Adam_1/readIdentityDiscriminator/out/kernel/Adam_1*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�
�
-Discriminator/out/bias/Adam/Initializer/zerosConst*
_output_shapes
:*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0
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
"Discriminator/out/bias/Adam/AssignAssignDiscriminator/out/bias/Adam-Discriminator/out/bias/Adam/Initializer/zeros*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
 Discriminator/out/bias/Adam/readIdentityDiscriminator/out/bias/Adam*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
�
/Discriminator/out/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*)
_class
loc:@Discriminator/out/bias*
valueB*    
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
"Discriminator/out/bias/Adam_1/readIdentityDiscriminator/out/bias/Adam_1*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
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
Adam/beta2Adam/epsilongradients/AddN_8*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
UAdam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam	ApplyAdam?Discriminator/second/Discriminator/secondfully_connected/kernelDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_locking( *
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
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
Adam/beta2Adam/epsilongradients/AddN_1*
_output_shapes
:	�*
use_locking( *
T0*+
_class!
loc:@Discriminator/out/kernel*
use_nesterov( 
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
gradients_1/Mean_1_grad/ShapeShapeLog_2*
_output_shapes
:*
T0*
out_type0
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
gradients_1/sub_1_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
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
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*
_output_shapes
: 
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
Dgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/Discriminator/out_1/MatMul_grad/MatMul=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Fgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*G
_class=
;9loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	�*
T0
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
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Sgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
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
Jgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumMgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeJgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumOgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
agradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1X^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
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
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1bgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Tgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapePgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
lgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddNe^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
ngradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrade^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*r
_classh
fdloc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
Ygradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMullgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
kgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityYgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMuld^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*l
_classb
`^loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
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
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
Zgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Kgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualkgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Mgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeroskgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Hgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumKgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectZgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1V^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
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
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1`gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
jgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1c^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
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
Cgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
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
@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/Generator/dense/MatMul_grad/MatMul9^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/Generator/dense/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Bgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/Generator/dense/MatMul_grad/MatMul_19^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/Generator/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeShape-Generator/forth/Generator/forthleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
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
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zerosFillBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
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
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1SumCgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1Rgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Kgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeE^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeL^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Ugradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1L^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
�
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Tgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulMulSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumSumBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulTgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
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
Hgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
gradients_1/AddN_2AddNUgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
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
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_2fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_2hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/forth/Generator/forthfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*(
_output_shapes
:����������*
T0
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
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
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:�
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
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:�*
T0
�
Sgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Xgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad
�
`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
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
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/forth/Generator/forthbatch_normalized/moving_mean/read*
T0*
_output_shapes	
:�
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
_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*`
_classV
TRloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul
�
agradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*b
_classX
VTloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
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
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_3?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
�
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1
�
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeShape-Generator/third/Generator/thirdleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Shape_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zerosFillBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Ggradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqualGreaterEqual-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Pgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Agradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectSelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Cgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1SelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/SumSumAgradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectPgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
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
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulMulSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumSumBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulTgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
gradients_1/AddN_4AddNUgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������*
T0
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:�*
dtype0
�
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_4fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
�
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape
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
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAddigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
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
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1
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
`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
bgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*f
_class\
ZXloc:@gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad
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
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Mgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/third/Generator/thirdfully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:�*
T0
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
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/zerosFillDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:����������*
T0
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
Egradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1SelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumSumCgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectRgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeReshape@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1SumEgradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1Tgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Fgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1ReshapeBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
�
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
Vgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Ygradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeR^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*[
_classQ
OMloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape
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
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_6hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
�
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
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
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumSumVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mulhgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAddkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1Mulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1BGenerator/second/Generator/secondbatch_normalized/moving_mean/read*
T0*
_output_shapes	
:�
�
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulY^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Muld^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�*
T0
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
agradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulZ^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
cgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1Z^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_7AddNmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N
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
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
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
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosFillBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:����������*
T0
�
Ggradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualGreaterEqual-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Pgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Agradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectSelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Cgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1SelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqual@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
>gradients_1/Generator/first/Generator/firstleaky_relu_grad/SumSumAgradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectPgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeReshape>gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1SumCgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1Rgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Dgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Kgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeE^gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeL^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*U
_classK
IGloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape*(
_output_shapes
:����������*
T0
�
Ugradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1L^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Mul/Generator/first/Generator/firstleaky_relu/alphaSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*[
_classQ
OMloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1
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
`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8Y^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
bgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
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
_gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*`
_classV
TRloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d*
T0
�
agradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d�
�
beta1_power_1/initial_valueConst*
dtype0*
_output_shapes
: *'
_class
loc:@Generator/dense/bias*
valueB
 *fff?
�
beta1_power_1
VariableV2*
_output_shapes
: *
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape: *
dtype0
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
VariableV2*
_output_shapes
: *
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape: *
dtype0
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
w
beta2_power_1/readIdentitybeta2_power_1*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
�
\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   �   *
dtype0
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
VariableV2*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name 
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
?Generator/first/Generator/firstfully_connected/kernel/Adam/readIdentity:Generator/first/Generator/firstfully_connected/kernel/Adam*
_output_shapes
:	d�*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
�
^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   �   
�
TGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0
�
NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	d�*
T0
�
<Generator/first/Generator/firstfully_connected/kernel/Adam_1
VariableV2*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d�*
dtype0*
_output_shapes
:	d�*
shared_name 
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
AGenerator/first/Generator/firstfully_connected/kernel/Adam_1/readIdentity<Generator/first/Generator/firstfully_connected/kernel/Adam_1*
_output_shapes
:	d�*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
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
?Generator/first/Generator/firstfully_connected/bias/Adam/AssignAssign8Generator/first/Generator/firstfully_connected/bias/AdamJGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
=Generator/first/Generator/firstfully_connected/bias/Adam/readIdentity8Generator/first/Generator/firstfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:�
�
LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB�*    *
dtype0
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
?Generator/first/Generator/firstfully_connected/bias/Adam_1/readIdentity:Generator/first/Generator/firstfully_connected/bias/Adam_1*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
�
^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
TGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    *
dtype0
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
�
CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignAssign<Generator/second/Generator/secondfully_connected/kernel/AdamNGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(
�
AGenerator/second/Generator/secondfully_connected/kernel/Adam/readIdentity<Generator/second/Generator/secondfully_connected/kernel/Adam* 
_output_shapes
:
��*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
�
`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
�
VGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    
�
PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zerosFill`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorVGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
�
>Generator/second/Generator/secondfully_connected/kernel/Adam_1
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
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
CGenerator/second/Generator/secondfully_connected/kernel/Adam_1/readIdentity>Generator/second/Generator/secondfully_connected/kernel/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
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
NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
<Generator/second/Generator/secondfully_connected/bias/Adam_1
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
�
CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignAssign<Generator/second/Generator/secondfully_connected/bias/Adam_1NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
�
AGenerator/second/Generator/secondfully_connected/bias/Adam_1/readIdentity<Generator/second/Generator/secondfully_connected/bias/Adam_1*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:�
�
NGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB�*    
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
EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignAssign>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/readIdentity>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:�
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
BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignAssign;Generator/second/Generator/secondbatch_normalized/beta/AdamMGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(
�
@Generator/second/Generator/secondbatch_normalized/beta/Adam/readIdentity;Generator/second/Generator/secondbatch_normalized/beta/Adam*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:�
�
OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB�*    *
dtype0
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
DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignAssign=Generator/second/Generator/secondbatch_normalized/beta/Adam_1OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(
�
BGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/readIdentity=Generator/second/Generator/secondbatch_normalized/beta/Adam_1*
_output_shapes	
:�*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
�
\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
RGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0
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
^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      
�
TGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0
�
NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0
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
AGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/readIdentity<Generator/third/Generator/thirdfully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container 
�
?Generator/third/Generator/thirdfully_connected/bias/Adam/AssignAssign8Generator/third/Generator/thirdfully_connected/bias/AdamJGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(
�
=Generator/third/Generator/thirdfully_connected/bias/Adam/readIdentity8Generator/third/Generator/thirdfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:�
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
VariableV2*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
?Generator/third/Generator/thirdfully_connected/bias/Adam_1/readIdentity:Generator/third/Generator/thirdfully_connected/bias/Adam_1*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
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
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAssign:Generator/third/Generator/thirdbatch_normalized/gamma/AdamLGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignAssign<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/readIdentity<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
�
KGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zerosConst*
_output_shapes	
:�*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB�*    *
dtype0
�
9Generator/third/Generator/thirdbatch_normalized/beta/Adam
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
@Generator/third/Generator/thirdbatch_normalized/beta/Adam/AssignAssign9Generator/third/Generator/thirdbatch_normalized/beta/AdamKGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
>Generator/third/Generator/thirdbatch_normalized/beta/Adam/readIdentity9Generator/third/Generator/thirdbatch_normalized/beta/Adam*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:�
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
BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignAssign;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
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
AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAssign:Generator/forth/Generator/forthfully_connected/kernel/AdamLGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
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
PGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0
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
?Generator/forth/Generator/forthfully_connected/bias/Adam/AssignAssign8Generator/forth/Generator/forthfully_connected/bias/AdamJGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
=Generator/forth/Generator/forthfully_connected/bias/Adam/readIdentity8Generator/forth/Generator/forthfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:�
�
\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�
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
LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/Const*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0
�
:Generator/forth/Generator/forthfully_connected/bias/Adam_1
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
AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAssign:Generator/forth/Generator/forthfully_connected/bias/Adam_1LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(
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
RGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
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
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAssign:Generator/forth/Generator/forthbatch_normalized/gamma/AdamLGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
TGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    
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
CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignAssign<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/readIdentity<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:�*
T0
�
[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:�*
dtype0*
_output_shapes
:
�
QGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
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
@Generator/forth/Generator/forthbatch_normalized/beta/Adam/AssignAssign9Generator/forth/Generator/forthbatch_normalized/beta/AdamKGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
>Generator/forth/Generator/forthbatch_normalized/beta/Adam/readIdentity9Generator/forth/Generator/forthbatch_normalized/beta/Adam*
_output_shapes	
:�*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
�
]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:�*
dtype0
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
 Generator/dense/kernel/Adam/readIdentityGenerator/dense/kernel/Adam*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
��
�
?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
5Generator/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *)
_class
loc:@Generator/dense/kernel*
valueB
 *    
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
VariableV2*)
_class
loc:@Generator/dense/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
$Generator/dense/kernel/Adam_1/AssignAssignGenerator/dense/kernel/Adam_1/Generator/dense/kernel/Adam_1/Initializer/zeros*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *'
_class
loc:@Generator/dense/bias*
	container 
�
 Generator/dense/bias/Adam/AssignAssignGenerator/dense/bias/Adam+Generator/dense/bias/Adam/Initializer/zeros*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
Generator/dense/bias/Adam/readIdentityGenerator/dense/bias/Adam*
_output_shapes	
:�*
T0*'
_class
loc:@Generator/dense/bias
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
"Generator/dense/bias/Adam_1/AssignAssignGenerator/dense/bias/Adam_1-Generator/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:�
�
 Generator/dense/bias/Adam_1/readIdentityGenerator/dense/bias/Adam_1*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:�*
T0
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
Adam_1/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
S
Adam_1/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
MAdam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/first/Generator/firstfully_connected/kernel:Generator/first/Generator/firstfully_connected/kernel/Adam<Generator/first/Generator/firstfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:	d�*
use_locking( *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
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
MAdam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdam	ApplyAdam5Generator/second/Generator/secondfully_connected/bias:Generator/second/Generator/secondfully_connected/bias/Adam<Generator/second/Generator/secondfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
�
OAdam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdam	ApplyAdam7Generator/second/Generator/secondbatch_normalized/gamma<Generator/second/Generator/secondbatch_normalized/gamma/Adam>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:�*
use_locking( *
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
use_nesterov( 
�
NAdam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdam	ApplyAdam6Generator/second/Generator/secondbatch_normalized/beta;Generator/second/Generator/secondbatch_normalized/beta/Adam=Generator/second/Generator/secondbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
�
MAdam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdfully_connected/kernel:Generator/third/Generator/thirdfully_connected/kernel/Adam<Generator/third/Generator/thirdfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
use_nesterov( 
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
LAdam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/third/Generator/thirdbatch_normalized/beta9Generator/third/Generator/thirdbatch_normalized/beta/Adam;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:�
�
MAdam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthfully_connected/kernel:Generator/forth/Generator/forthfully_connected/kernel/Adam<Generator/forth/Generator/forthfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
use_locking( *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
use_nesterov( 
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
.Adam_1/update_Generator/dense/kernel/ApplyAdam	ApplyAdamGenerator/dense/kernelGenerator/dense/kernel/AdamGenerator/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@Generator/dense/kernel*
use_nesterov( * 
_output_shapes
:
��
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


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*'
_class
loc:@Generator/dense/bias
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
_output_shapes
: *
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(
�

Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*'
_class
loc:@Generator/dense/bias*
_output_shapes
: *
T0
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*'
_class
loc:@Generator/dense/bias
�	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: "��n�b     ��x6	T�3֩��AJ��
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
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/sub*
_output_shapes
:	d�*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
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
dtype0*
_output_shapes
:	d�*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d�
�
<Generator/first/Generator/firstfully_connected/kernel/AssignAssign5Generator/first/Generator/firstfully_connected/kernelPGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	d�*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
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
5Generator/first/Generator/firstfully_connected/MatMulMatMulGenerator/noise:Generator/first/Generator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
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
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/subSubVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
�
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulMul`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
��*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
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
VariableV2*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
�
>Generator/second/Generator/secondfully_connected/kernel/AssignAssign7Generator/second/Generator/secondfully_connected/kernelRGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
��*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(
�
<Generator/second/Generator/secondfully_connected/kernel/readIdentity7Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
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
<Generator/second/Generator/secondfully_connected/bias/AssignAssign5Generator/second/Generator/secondfully_connected/biasGGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
:Generator/second/Generator/secondfully_connected/bias/readIdentity5Generator/second/Generator/secondfully_connected/bias*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:�
�
7Generator/second/Generator/secondfully_connected/MatMulMatMul)Generator/first/Generator/firstleaky_relu<Generator/second/Generator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
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
>Generator/second/Generator/secondbatch_normalized/gamma/AssignAssign7Generator/second/Generator/secondbatch_normalized/gammaHGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones*
_output_shapes	
:�*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(
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
=Generator/second/Generator/secondbatch_normalized/beta/AssignAssign6Generator/second/Generator/secondbatch_normalized/betaHGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignAssign=Generator/second/Generator/secondbatch_normalized/moving_meanOGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:�
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
VariableV2*
_output_shapes	
:�*
shared_name *T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
	container *
shape:�*
dtype0
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
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
?Generator/second/Generator/secondbatch_normalized/batchnorm/addAddFGenerator/second/Generator/secondbatch_normalized/moving_variance/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/add/y*
_output_shapes	
:�*
T0
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/RsqrtRsqrt?Generator/second/Generator/secondbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:�
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
?Generator/second/Generator/secondbatch_normalized/batchnorm/subSub;Generator/second/Generator/secondbatch_normalized/beta/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1AddAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1?Generator/second/Generator/secondbatch_normalized/batchnorm/sub*(
_output_shapes
:����������*
T0
v
1Generator/second/Generator/secondleaky_relu/alphaConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
/Generator/second/Generator/secondleaky_relu/mulMul1Generator/second/Generator/secondleaky_relu/alphaAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
+Generator/second/Generator/secondleaky_reluMaximum/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
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
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
_output_shapes
: *
T0
�
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��
�
PGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniformAddTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
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
8Generator/third/Generator/thirdfully_connected/bias/readIdentity3Generator/third/Generator/thirdfully_connected/bias*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:�
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
<Generator/third/Generator/thirdbatch_normalized/gamma/AssignAssign5Generator/third/Generator/thirdbatch_normalized/gammaFGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
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
VariableV2*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
;Generator/third/Generator/thirdbatch_normalized/beta/AssignAssign4Generator/third/Generator/thirdbatch_normalized/betaFGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
�
9Generator/third/Generator/thirdbatch_normalized/beta/readIdentity4Generator/third/Generator/thirdbatch_normalized/beta*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:�
�
MGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zerosConst*
_output_shapes	
:�*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
valueB�*    *
dtype0
�
;Generator/third/Generator/thirdbatch_normalized/moving_mean
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean
�
BGenerator/third/Generator/thirdbatch_normalized/moving_mean/AssignAssign;Generator/third/Generator/thirdbatch_normalized/moving_meanMGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
validate_shape(
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
FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignAssign?Generator/third/Generator/thirdbatch_normalized/moving_variancePGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones*
_output_shapes	
:�*
use_locking(*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
validate_shape(
�
DGenerator/third/Generator/thirdbatch_normalized/moving_variance/readIdentity?Generator/third/Generator/thirdbatch_normalized/moving_variance*
_output_shapes	
:�*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
=Generator/third/Generator/thirdbatch_normalized/batchnorm/addAddDGenerator/third/Generator/thirdbatch_normalized/moving_variance/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/y*
_output_shapes	
:�*
T0
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
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  �=*
dtype0
�
^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
seed2m*
dtype0* 
_output_shapes
:
��*

seed
�
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
_output_shapes
: 
�
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/sub*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��*
T0
�
PGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniformAddTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��*
T0
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
:Generator/forth/Generator/forthfully_connected/kernel/readIdentity5Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
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
5Generator/forth/Generator/forthfully_connected/MatMulMatMul)Generator/third/Generator/thirdleaky_relu:Generator/forth/Generator/forthfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
6Generator/forth/Generator/forthfully_connected/BiasAddBiasAdd5Generator/forth/Generator/forthfully_connected/MatMul8Generator/forth/Generator/forthfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
�
VGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:�
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
<Generator/forth/Generator/forthbatch_normalized/gamma/AssignAssign5Generator/forth/Generator/forthbatch_normalized/gammaFGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(
�
:Generator/forth/Generator/forthbatch_normalized/gamma/readIdentity5Generator/forth/Generator/forthbatch_normalized/gamma*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:�
�
VGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:�
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
FGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zerosFillVGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/Const*
_output_shapes	
:�*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
	container *
shape:�
�
BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignAssign;Generator/forth/Generator/forthbatch_normalized/moving_meanMGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
@Generator/forth/Generator/forthbatch_normalized/moving_mean/readIdentity;Generator/forth/Generator/forthbatch_normalized/moving_mean*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
_output_shapes	
:�
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
PGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/onesFill`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorVGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/Const*
_output_shapes	
:�*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*

index_type0
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
FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignAssign?Generator/forth/Generator/forthbatch_normalized/moving_variancePGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
/Generator/forth/Generator/forthleaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
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
7Generator/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0
�
5Generator/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *)
_class
loc:@Generator/dense/kernel*
valueB
 *z�k�*
dtype0
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
seed2�*
dtype0* 
_output_shapes
:
��*

seed*
T0*)
_class
loc:@Generator/dense/kernel
�
5Generator/dense/kernel/Initializer/random_uniform/subSub5Generator/dense/kernel/Initializer/random_uniform/max5Generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@Generator/dense/kernel*
_output_shapes
: 
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
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *)
_class
loc:@Generator/dense/kernel
�
Generator/dense/kernel/AssignAssignGenerator/dense/kernel1Generator/dense/kernel/Initializer/random_uniform*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
Generator/dense/kernel/readIdentityGenerator/dense/kernel*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
��
�
&Generator/dense/bias/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Generator/dense/bias
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
Generator/dense/bias/AssignAssignGenerator/dense/bias&Generator/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:�
�
Generator/dense/bias/readIdentityGenerator/dense/bias*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:�
�
Generator/dense/MatMulMatMul)Generator/forth/Generator/forthleaky_reluGenerator/dense/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
Generator/dense/BiasAddBiasAddGenerator/dense/MatMulGenerator/dense/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
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
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HY��
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

seed*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
seed2�*
dtype0* 
_output_shapes
:
��
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/subSub\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/max\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mulMulfDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniform\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/sub*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
�
XDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniformAdd\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mul\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
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
DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignAssign=Discriminator/first/Discriminator/firstfully_connected/kernelXDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignAssign;Discriminator/first/Discriminator/firstfully_connected/biasMDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
@Discriminator/first/Discriminator/firstfully_connected/bias/readIdentity;Discriminator/first/Discriminator/firstfully_connected/bias*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:�
�
=Discriminator/first/Discriminator/firstfully_connected/MatMulMatMulDiscriminator/realBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
>Discriminator/first/Discriminator/firstfully_connected/BiasAddBiasAdd=Discriminator/first/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
|
7Discriminator/first/Discriminator/firstleaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
�
5Discriminator/first/Discriminator/firstleaky_relu/mulMul7Discriminator/first/Discriminator/firstleaky_relu/alpha>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
1Discriminator/first/Discriminator/firstleaky_reluMaximum5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      
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
ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniformAdd^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mul^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��*
T0
�
?Discriminator/second/Discriminator/secondfully_connected/kernel
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
FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignAssign?Discriminator/second/Discriminator/secondfully_connected/kernelZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��
�
DDiscriminator/second/Discriminator/secondfully_connected/kernel/readIdentity?Discriminator/second/Discriminator/secondfully_connected/kernel*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��
�
ODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB�*    
�
=Discriminator/second/Discriminator/secondfully_connected/bias
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias
�
DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignAssign=Discriminator/second/Discriminator/secondfully_connected/biasODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
BDiscriminator/second/Discriminator/secondfully_connected/bias/readIdentity=Discriminator/second/Discriminator/secondfully_connected/bias*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:�
�
?Discriminator/second/Discriminator/secondfully_connected/MatMulMatMul1Discriminator/first/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
@Discriminator/second/Discriminator/secondfully_connected/BiasAddBiasAdd?Discriminator/second/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
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
ADiscriminator/out/kernel/Initializer/random_uniform/RandomUniformRandomUniform9Discriminator/out/kernel/Initializer/random_uniform/shape*+
_class!
loc:@Discriminator/out/kernel*
seed2�*
dtype0*
_output_shapes
:	�*

seed*
T0
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
3Discriminator/out/kernel/Initializer/random_uniformAdd7Discriminator/out/kernel/Initializer/random_uniform/mul7Discriminator/out/kernel/Initializer/random_uniform/min*
_output_shapes
:	�*
T0*+
_class!
loc:@Discriminator/out/kernel
�
Discriminator/out/kernel
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	�
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
Discriminator/out/kernel/readIdentityDiscriminator/out/kernel*
_output_shapes
:	�*
T0*+
_class!
loc:@Discriminator/out/kernel
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
VariableV2*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
Discriminator/out/bias/AssignAssignDiscriminator/out/bias(Discriminator/out/bias/Initializer/zeros*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
Discriminator/out/bias/readIdentityDiscriminator/out/bias*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
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
?Discriminator/first_1/Discriminator/firstfully_connected/MatMulMatMulGenerator/TanhBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
�
@Discriminator/first_1/Discriminator/firstfully_connected/BiasAddBiasAdd?Discriminator/first_1/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
~
9Discriminator/first_1/Discriminator/firstleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *��L>*
dtype0
�
7Discriminator/first_1/Discriminator/firstleaky_relu/mulMul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
3Discriminator/first_1/Discriminator/firstleaky_reluMaximum7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
ADiscriminator/second_1/Discriminator/secondfully_connected/MatMulMatMul3Discriminator/first_1/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
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
9Discriminator/second_1/Discriminator/secondleaky_relu/mulMul;Discriminator/second_1/Discriminator/secondleaky_relu/alphaBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
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
Log_1Logsub*'
_output_shapes
:���������*
T0
H
addAddLogLog_1*'
_output_shapes
:���������*
T0
V
ConstConst*
dtype0*
_output_shapes
:*
valueB"       
V
MeanMeanaddConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
1
NegNegMean*
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
discriminator_lossHistogramSummarydiscriminator_loss/tagNeg*
_output_shapes
: *
T0
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
gradients/Mean_grad/Shape_1Shapeadd*
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
gradients/add_grad/ShapeShapeLog*
out_type0*
_output_shapes
:*
T0
_
gradients/add_grad/Shape_1ShapeLog_1*
_output_shapes
:*
T0*
out_type0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/add_grad/SumSumgradients/Mean_grad/truediv(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
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
ReciprocalDiscriminator/out/Sigmoid,^gradients/add_grad/tuple/control_dependency*
T0*'
_output_shapes
:���������
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
gradients/Log_1_grad/mulMul-gradients/add_grad/tuple/control_dependency_1gradients/Log_1_grad/Reciprocal*'
_output_shapes
:���������*
T0
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
gradients/sub_grad/Shape_1ShapeDiscriminator/out_1/Sigmoid*
_output_shapes
:*
T0*
out_type0
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSumgradients/Log_1_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
4gradients/Discriminator/out/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
9gradients/Discriminator/out/BiasAdd_grad/tuple/group_depsNoOp5^gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad5^gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad
�
Agradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyIdentity4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:���������*
T0
�
Cgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad
�
6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid-gradients/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
.gradients/Discriminator/out/MatMul_grad/MatMulMatMulAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
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
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zerosFillJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
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
Lgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Sgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeM^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1
�
[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeT^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*]
_classS
QOloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape
�
]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1T^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
0gradients/Discriminator/out_1/MatMul_grad/MatMulMatMulCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/MulMul[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumSumJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul9Discriminator/second/Discriminator/secondleaky_relu/alpha[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Pgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*c
_classY
WUloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1
�
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
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
Mgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumMgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1\gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
gradients/AddN_2AddN]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1
�
[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:�
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
jgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
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
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1`gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Rgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ygradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeS^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
�
agradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeZ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*c
_classY
WUloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
�
cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1Z^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Ugradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
ggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*h
_class^
\Zloc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
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
]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
_output_shapes	
:�*
T0*
data_formatNHWC
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
lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradc^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*p
_classf
dbloc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
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
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Shapeggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
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
Mgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Vgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Ggradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectSelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
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
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeReshapeDgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
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
Wgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMuljgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Ygradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relujgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
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
kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1b^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*l
_classb
`^loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
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
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Zgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulMulYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumSumHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulZgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1\gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ngradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Ogradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Xgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumKgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1Zgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
N*(
_output_shapes
:����������*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1
�
Ygradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*
data_formatNHWC*
_output_shapes	
:�
�
^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6Z^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
fgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1
�
hgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
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
\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Pgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
egradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentitySgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*f
_class\
ZXloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul
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
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
data_formatNHWC*
_output_shapes	
:�*
T0
�
`gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_7\^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
hgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_7a^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
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
igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
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
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
�
beta2_power/readIdentitybeta2_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
�
dDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
�
ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0
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
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container 
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
GDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam* 
_output_shapes
:
��*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
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
VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zerosFillfDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0
�
DDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
��*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
��
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
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/readIdentityDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
�
RDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
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
GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignAssign@Discriminator/first/Discriminator/firstfully_connected/bias/AdamRDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
VariableV2*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
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
^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
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
VariableV2*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��
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
VariableV2*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:�*
dtype0
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
dtype0*
_output_shapes
:	�*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	�
�
$Discriminator/out/kernel/Adam/AssignAssignDiscriminator/out/kernel/Adam/Discriminator/out/kernel/Adam/Initializer/zeros*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0
�
"Discriminator/out/kernel/Adam/readIdentityDiscriminator/out/kernel/Adam*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	�*
T0
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
VariableV2*
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container 
�
&Discriminator/out/kernel/Adam_1/AssignAssignDiscriminator/out/kernel/Adam_11Discriminator/out/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	�
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
VariableV2*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
"Discriminator/out/bias/Adam/AssignAssignDiscriminator/out/bias/Adam-Discriminator/out/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias
�
 Discriminator/out/bias/Adam/readIdentityDiscriminator/out/bias/Adam*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:*
T0
�
/Discriminator/out/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0
�
Discriminator/out/bias/Adam_1
VariableV2*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container *
shape:*
dtype0*
_output_shapes
:
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
Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
SAdam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam	ApplyAdam=Discriminator/first/Discriminator/firstfully_connected/kernelBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
�
QAdam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdam	ApplyAdam;Discriminator/first/Discriminator/firstfully_connected/bias@Discriminator/first/Discriminator/firstfully_connected/bias/AdamBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:�
�
UAdam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam	ApplyAdam?Discriminator/second/Discriminator/secondfully_connected/kernelDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_locking( *
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��
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
Adam/beta2Adam/epsilongradients/AddN_1*
use_nesterov( *
_output_shapes
:	�*
use_locking( *
T0*+
_class!
loc:@Discriminator/out/kernel
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
Adam/beta2R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
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
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
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
gradients_1/sub_1_grad/SumSumgradients_1/Log_2_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
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
Ggradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad
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
Fgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*G
_class=
;9loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1
�
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
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
Sgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
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
Ogradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
agradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1X^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1bgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddNAddNagradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N
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
ngradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrade^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*r
_classh
fdloc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
Ygradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMullgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
kgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityYgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMuld^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
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
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapekgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
�
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Qgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
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
Mgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeroskgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumMgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1\gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ugradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpM^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeO^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeV^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape
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
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
agradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeZ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape
�
cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1Z^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_1AddN_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N
�
]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
_output_shapes	
:�*
T0*
data_formatNHWC
�
bgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1^^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
jgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1c^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
lgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradc^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*p
_classf
dbloc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
Wgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMuljgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
�
Ygradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhjgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
�
agradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulZ^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
�
igradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulb^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*j
_class`
^\loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
kgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1b^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
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
Agradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/Generator/Tanh_grad/TanhGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*;
_class1
/-loc:@gradients_1/Generator/Tanh_grad/TanhGrad
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
Bgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/Generator/dense/MatMul_grad/MatMul_19^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/Generator/dense/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeShape-Generator/forth/Generator/forthleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
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
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zerosFillBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:����������*
T0
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
Cgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1SelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/SumSumAgradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectPgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeReshape>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1SumCgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1Rgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Kgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeE^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeL^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
Ugradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1L^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
�
Tgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulMulSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumSumBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulTgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Mul/Generator/forth/Generator/forthleaky_relu/alphaSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
gradients_1/AddN_2AddNUgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
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
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_2fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
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
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAddigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�*
T0
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
`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������
�
bgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*f
_class\
ZXloc:@gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
_output_shapes	
:�*
T0
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/forth/Generator/forthbatch_normalized/moving_mean/read*
T0*
_output_shapes	
:�
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
agradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*b
_classX
VTloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
gradients_1/AddN_3AddNkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
N*
_output_shapes	
:�*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
�
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_3:Generator/forth/Generator/forthbatch_normalized/gamma/read*
_output_shapes	
:�*
T0
�
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_3?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:�
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
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zerosFillBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Ggradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqualGreaterEqual-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Pgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeReshape>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
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
Ugradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1L^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1
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
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulMulSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
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
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Hgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Ogradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1
�
Wgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape
�
Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
gradients_1/AddN_4AddNUgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������*
T0
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
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
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
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
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�*
T0
�
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg
�
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg
�
Sgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:�
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
bgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*f
_class\
ZXloc:@gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad
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
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul
�
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*i
_class_
][loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1
�
Mgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/third/Generator/thirdfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
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
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�*
T0
�
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeShape/Generator/second/Generator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Shape_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/zerosFillDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
�
Igradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualGreaterEqual/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Rgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumSumCgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectRgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeReshape@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1SumEgradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1Tgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Mul1Generator/second/Generator/secondleaky_relu/alphaUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Xgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
gradients_1/AddN_6AddNWgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������*
T0
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1*
out_type0*
_output_shapes
:*
T0
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
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
�
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:����������
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
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*(
_output_shapes
:����������*
T0
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
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/NegNegmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
�
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpn^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1U^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg
�
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitymgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:�*
T0
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
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*k
_classa
_]loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1
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
agradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulZ^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������
�
cgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1Z^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_7AddNmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
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
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeShape-Generator/first/Generator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Shapeagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0
�
Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
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
Pgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Agradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectSelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
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
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeReshape>gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1SumCgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1Rgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Dgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
Ugradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1L^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
�
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
Tgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulMulSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumSumBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulTgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Mul/Generator/first/Generator/firstleaky_relu/alphaSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Hgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
�
Ogradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1
�
Wgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
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
`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8Y^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
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
Ogradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d�*
transpose_a(*
transpose_b( 
�
Wgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1
�
_gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul*'
_output_shapes
:���������d
�
agradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d�
�
beta1_power_1/initial_valueConst*'
_class
loc:@Generator/dense/bias*
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
shared_name *'
_class
loc:@Generator/dense/bias*
	container 
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
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@Generator/dense/bias*
	container 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
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
LGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zerosFill\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	d�*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0
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
?Generator/first/Generator/firstfully_connected/kernel/Adam/readIdentity:Generator/first/Generator/firstfully_connected/kernel/Adam*
_output_shapes
:	d�*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
�
^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
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
?Generator/first/Generator/firstfully_connected/bias/Adam/AssignAssign8Generator/first/Generator/firstfully_connected/bias/AdamJGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
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
_output_shapes
:
��*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
��*
dtype0
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
AGenerator/second/Generator/secondfully_connected/kernel/Adam/readIdentity<Generator/second/Generator/secondfully_connected/kernel/Adam* 
_output_shapes
:
��*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
�
`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      
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
�
EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignAssign>Generator/second/Generator/secondfully_connected/kernel/Adam_1PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
CGenerator/second/Generator/secondfully_connected/kernel/Adam_1/readIdentity>Generator/second/Generator/secondfully_connected/kernel/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
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
AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAssign:Generator/second/Generator/secondfully_connected/bias/AdamLGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
?Generator/second/Generator/secondfully_connected/bias/Adam/readIdentity:Generator/second/Generator/secondfully_connected/bias/Adam*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
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
CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignAssign<Generator/second/Generator/secondfully_connected/bias/Adam_1NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
AGenerator/second/Generator/secondfully_connected/bias/Adam_1/readIdentity<Generator/second/Generator/secondfully_connected/bias/Adam_1*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:�
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
EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignAssign>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/readIdentity>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:�
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
BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignAssign;Generator/second/Generator/secondbatch_normalized/beta/AdamMGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(
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
RGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0
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
AGenerator/third/Generator/thirdfully_connected/kernel/Adam/AssignAssign:Generator/third/Generator/thirdfully_connected/kernel/AdamLGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
?Generator/third/Generator/thirdfully_connected/kernel/Adam/readIdentity:Generator/third/Generator/thirdfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��
�
^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      
�
TGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0
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
CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignAssign<Generator/third/Generator/thirdfully_connected/kernel/Adam_1NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
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
?Generator/third/Generator/thirdfully_connected/bias/Adam/AssignAssign8Generator/third/Generator/thirdfully_connected/bias/AdamJGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:�
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
VariableV2*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
?Generator/third/Generator/thirdbatch_normalized/gamma/Adam/readIdentity:Generator/third/Generator/thirdbatch_normalized/gamma/Adam*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
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
CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignAssign<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container 
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
>Generator/third/Generator/thirdbatch_normalized/beta/Adam/readIdentity9Generator/third/Generator/thirdbatch_normalized/beta/Adam*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:�*
T0
�
MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB�*    *
dtype0
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
AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAssign:Generator/forth/Generator/forthfully_connected/kernel/AdamLGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
?Generator/forth/Generator/forthfully_connected/kernel/Adam/readIdentity:Generator/forth/Generator/forthfully_connected/kernel/Adam* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
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
NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0
�
<Generator/forth/Generator/forthfully_connected/kernel/Adam_1
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
CGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/AssignAssign<Generator/forth/Generator/forthfully_connected/kernel/Adam_1NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros* 
_output_shapes
:
��*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(
�
AGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/readIdentity<Generator/forth/Generator/forthfully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
�
ZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�*
dtype0
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
?Generator/forth/Generator/forthfully_connected/bias/Adam/AssignAssign8Generator/forth/Generator/forthfully_connected/bias/AdamJGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�
�
=Generator/forth/Generator/forthfully_connected/bias/Adam/readIdentity8Generator/forth/Generator/forthfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:�
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
LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/Const*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:�*
T0
�
:Generator/forth/Generator/forthfully_connected/bias/Adam_1
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
AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAssign:Generator/forth/Generator/forthfully_connected/bias/Adam_1LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
?Generator/forth/Generator/forthfully_connected/bias/Adam_1/readIdentity:Generator/forth/Generator/forthfully_connected/bias/Adam_1*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:�*
T0
�
\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:�*
dtype0*
_output_shapes
:
�
RGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 
�
LGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zerosFill\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/Const*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:�*
T0
�
:Generator/forth/Generator/forthbatch_normalized/gamma/Adam
VariableV2*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:�*
dtype0
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
NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/Const*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:�*
T0
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
KGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zerosFill[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorQGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/Const*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:�*
T0
�
9Generator/forth/Generator/forthbatch_normalized/beta/Adam
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container 
�
@Generator/forth/Generator/forthbatch_normalized/beta/Adam/AssignAssign9Generator/forth/Generator/forthbatch_normalized/beta/AdamKGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:�
�
;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1
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
VariableV2*)
_class
loc:@Generator/dense/kernel*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name 
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
/Generator/dense/kernel/Adam_1/Initializer/zerosFill?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5Generator/dense/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*)
_class
loc:@Generator/dense/kernel*

index_type0
�
Generator/dense/kernel/Adam_1
VariableV2*
	container *
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *)
_class
loc:@Generator/dense/kernel
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
"Generator/dense/kernel/Adam_1/readIdentityGenerator/dense/kernel/Adam_1* 
_output_shapes
:
��*
T0*)
_class
loc:@Generator/dense/kernel
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *'
_class
loc:@Generator/dense/bias
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
Generator/dense/bias/Adam/readIdentityGenerator/dense/bias/Adam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:�
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
"Generator/dense/bias/Adam_1/AssignAssignGenerator/dense/bias/Adam_1-Generator/dense/bias/Adam_1/Initializer/zeros*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
 Generator/dense/bias/Adam_1/readIdentityGenerator/dense/bias/Adam_1*
_output_shapes	
:�*
T0*'
_class
loc:@Generator/dense/bias
Y
Adam_1/learning_rateConst*
_output_shapes
: *
valueB
 *�Q9*
dtype0
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
OAdam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdam	ApplyAdam7Generator/second/Generator/secondfully_connected/kernel<Generator/second/Generator/secondfully_connected/kernel/Adam>Generator/second/Generator/secondfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
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
OAdam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdam	ApplyAdam7Generator/second/Generator/secondbatch_normalized/gamma<Generator/second/Generator/secondbatch_normalized/gamma/Adam>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:�*
use_locking( *
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
use_nesterov( 
�
NAdam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdam	ApplyAdam6Generator/second/Generator/secondbatch_normalized/beta;Generator/second/Generator/secondbatch_normalized/beta/Adam=Generator/second/Generator/secondbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
MAdam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdfully_connected/kernel:Generator/third/Generator/thirdfully_connected/kernel/Adam<Generator/third/Generator/thirdfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
use_nesterov( 
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
MAdam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdbatch_normalized/gamma:Generator/third/Generator/thirdbatch_normalized/gamma/Adam<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
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
MAdam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthfully_connected/kernel:Generator/forth/Generator/forthfully_connected/kernel/Adam<Generator/forth/Generator/forthfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0
�
KAdam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdam	ApplyAdam3Generator/forth/Generator/forthfully_connected/bias8Generator/forth/Generator/forthfully_connected/bias/Adam:Generator/forth/Generator/forthfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
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
LAdam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/forth/Generator/forthbatch_normalized/beta9Generator/forth/Generator/forthbatch_normalized/beta/Adam;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:�
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
,Adam_1/update_Generator/dense/bias/ApplyAdam	ApplyAdamGenerator/dense/biasGenerator/dense/bias/AdamGenerator/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*'
_class
loc:@Generator/dense/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�

Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: ""��
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
Generator/dense/bias/Adam_1:0"Generator/dense/bias/Adam_1/Assign"Generator/dense/bias/Adam_1/read:02/Generator/dense/bias/Adam_1/Initializer/zeros:0"7
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
Adam_19d��       �N�	
�֩��A*�
w
discriminator_loss*a	    m#�?    m#�?      �?!    m#�?) H��� @23?��|�?�E̟���?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) @KG��?2uo�p�+Se*8��������:              �?        �ed��       �{�	���֩��A*�
w
discriminator_loss*a	   ����?   ����?      �?!   ����?) �BN�b�?2+Se*8�?uo�p�?�������:              �?        
s
generator_loss*a	   ��tۿ   ��tۿ      �?!   ��tۿ) ��;���?2��Z%�޿W�i�bۿ�������:              �?        ݁x�       �{�	f�֩��A
*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �@�ڸ?2_&A�o��?�Ca�G��?�������:              �?        
s
generator_loss*a	   ���Ϳ   ���Ϳ      �?!   ���Ϳ) �G�ɫ?2�Z�_��ο�K?̿�������:              �?        5�Cz�       �{�	@�֩��A*�
w
discriminator_loss*a	    ;�?    ;�?      �?!    ;�?) �k��?2!�����?Ӗ8��s�?�������:              �?        
s
generator_loss*a	   ��»�   ��»�      �?!   ��»�) �~���?2��(!�ؼ�%g�cE9���������:              �?        �8��       �{�	���֩��A*�
w
discriminator_loss*a	    D��?    D��?      �?!    D��?) ����g?2���g��?I���?�������:              �?        
s
generator_loss*a	   �Z��   �Z��      �?!   �Z��)����r
a?2�g���w���/�*>���������:              �?        w���       �{�	���֩��A*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) (�L?2��<�A��?�v��ab�?�������:              �?        
s
generator_loss*a	    �֕�    �֕�      �?!    �֕�) �p�S�=?2�"�uԖ�^�S�����������:              �?        *GY��       �{�	���֩��A*�
w
discriminator_loss*a	   ����?   ����?      �?!   ����?)�8���+?2�7c_XY�?�#�h/�?�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���) [}H?2>	� �����T}��������:              �?        )��       �{�	���֩��A#*�
w
discriminator_loss*a	    O��?    O��?      �?!    O��?) ��,T?2���J�\�?-Ա�L�?�������:              �?        
s
generator_loss*a	    �{�    �{�      �?!    �{�) 
��l�?2���T}�o��5sz��������:              �?        \gL�       �{�	���֩��A(*�
w
discriminator_loss*a	   �p��?   �p��?      �?!   �p��?) �Z*��?2>	� �?����=��?�������:              �?        
s
generator_loss*a	   @��q�   @��q�      �?!   @��q�) ��^��>2uWy��r�;8�clp��������:              �?        �a��       �{�	�N	ש��A-*�
w
discriminator_loss*a	   �(�u?   �(�u?      �?!   �(�u?)@l�kߖ�>2hyO�s?&b՞
�u?�������:              �?        
s
generator_loss*a	   �*!g�   �*!g�      �?!   �*!g�) r�з�>2P}���h�Tw��Nof��������:              �?        v��-�       �{�	N?ש��A2*�
w
discriminator_loss*a	   @R�q?   @R�q?      �?!   @R�q?) �Cм�>2;8�clp?uWy��r?�������:              �?        
s
generator_loss*a	    .�\�    .�\�      �?!    .�\�)  "A�>2E��{��^��m9�H�[��������:              �?        �7���       �{�	7:ש��A7*�
w
discriminator_loss*a	   ��;r?   ��;r?      �?!   ��;r?) �����>2uWy��r?hyO�s?�������:              �?        
s
generator_loss*a	   @c�`�   @c�`�      �?!   @c�`�) �g]��>2���%��b��l�P�`��������:              �?        �1-�       �{�	Kqש��A<*�
w
discriminator_loss*a	   ��o?   ��o?      �?!   ��o?)�Ĵ #��>2�N�W�m?;8�clp?�������:              �?        
s
generator_loss*a	   `ʀY�   `ʀY�      �?!   `ʀY�) ���BS�>2�m9�H�[���bB�SY��������:              �?        �Ŵ��       �{�	�{ש��AA*�
w
discriminator_loss*a	    !�j?    !�j?      �?!    !�j?) J�C�*�>2P}���h?ߤ�(g%k?�������:              �?        
s
generator_loss*a	   �bY�   �bY�      �?!   �bY�) ���v"�>2�m9�H�[���bB�SY��������:              �?        b����       �{�	΃ש��AF*�
w
discriminator_loss*a	   �AZc?   �AZc?      �?!   �AZc?) $�Xh�>2���%��b?5Ucv0ed?�������:              �?        
s
generator_loss*a	   �CJ�   �CJ�      �?!   �CJ�) �Ou��>2IcD���L��qU���I��������:              �?        {S\�       �{�	�!ש��AK*�
w
discriminator_loss*a	    ��c?    ��c?      �?!    ��c?)@ltp��>2���%��b?5Ucv0ed?�������:              �?        
s
generator_loss*a	   �� Q�   �� Q�      �?!   �� Q�) 2zV�>2nK���LQ�k�1^�sO��������:              �?        � �k�       �{�	܎%ש��AP*�
w
discriminator_loss*a	   �d�a?   �d�a?      �?!   �d�a?) Dg/	��>2�l�P�`?���%��b?�������:              �?        
s
generator_loss*a	   ��R�   ��R�      �?!   ��R�) �U��>2�lDZrS�nK���LQ��������:              �?        ��(��       �{�	k7Jש��AU*�
w
discriminator_loss*a	   �7f?   �7f?      �?!   �7f?)@ �/͡�>2Tw��Nof?P}���h?�������:              �?        
s
generator_loss*a	   �I"U�   �I"U�      �?!   �I"U�)@�:�J�>2ܗ�SsW�<DKc��T��������:              �?        dd��       �{�	� Nש��AZ*�
w
discriminator_loss*a	   �+�c?   �+�c?      �?!   �+�c?) DfS��>2���%��b?5Ucv0ed?�������:              �?        
s
generator_loss*a	    L�S�    L�S�      �?!    L�S�)  iq��>2<DKc��T��lDZrS��������:              �?        ��	��       �{�	Y"Rש��A_*�
w
discriminator_loss*a	   �/^?   �/^?      �?!   �/^?)��c��x�>2�m9�H�[?E��{��^?�������:              �?        
s
generator_loss*a	   @�SQ�   @�SQ�      �?!   @�SQ�) Y@�Ĳ>2�lDZrS�nK���LQ��������:              �?        X.���       �{�	�(Vש��Ad*�
w
discriminator_loss*a	   ��[?   ��[?      �?!   ��[?) �� -]�>2�m9�H�[?E��{��^?�������:              �?        
s
generator_loss*a	   ��N�   ��N�      �?!   ��N�) �.��>2k�1^�sO�IcD���L��������:              �?        2C���       �{�	�Zש��Ai*�
w
discriminator_loss*a	   �T'X?   �T'X?      �?!   �T'X?) "�/;�>2ܗ�SsW?��bB�SY?�������:              �?        
s
generator_loss*a	   `t-M�   `t-M�      �?!   `t-M�) 9�~���>2k�1^�sO�IcD���L��������:              �?        �� ~�       �{�	��]ש��An*�
w
discriminator_loss*a	   �!Y?   �!Y?      �?!   �!Y?) ����>2ܗ�SsW?��bB�SY?�������:              �?        
s
generator_loss*a	   `v�E�   `v�E�      �?!   `v�E�)@ʯ1DY�>2
����G�a�$��{E��������:              �?        ?�9�       �{�	98bש��As*�
w
discriminator_loss*a	    i0U?    i0U?      �?!    i0U?)@�rm��>2<DKc��T?ܗ�SsW?�������:              �?        
s
generator_loss*a	   @F�I�   @F�I�      �?!   @F�I�)�8�T�>2�qU���I�
����G��������:              �?        ��y�       �{�	/fש��Ax*�
w
discriminator_loss*a	    �U?    �U?      �?!    �U?) @ȑԻ>2<DKc��T?ܗ�SsW?�������:              �?        
s
generator_loss*a	   ��yG�   ��yG�      �?!   ��yG�)���8�>2
����G�a�$��{E��������:              �?        �d��       �{�	0��ש��A}*�
w
discriminator_loss*a	   ��Y?   ��Y?      �?!   ��Y?) O�ey��>2��bB�SY?�m9�H�[?�������:              �?        
s
generator_loss*a	    85L�    85L�      �?!    85L�)  �zݨ>2IcD���L��qU���I��������:              �?        �>Y��       b�D�	z��ש��A�*�
w
discriminator_loss*a	   `��Q?   `��Q?      �?!   `��Q?)@b
�p��>2nK���LQ?�lDZrS?�������:              �?        
s
generator_loss*a	   @��B�   @��B�      �?!   @��B�) ��b��>2�T���C��!�A��������:              �?        ��9A�       b�D�	�y�ש��A�*�
w
discriminator_loss*a	   ��P?   ��P?      �?!   ��P?) "��#�>2k�1^�sO?nK���LQ?�������:              �?        
s
generator_loss*a	   �F�C�   �F�C�      �?!   �F�C�) ��9�&�>2a�$��{E��T���C��������:              �?        �g6��       b�D�	���ש��A�*�
w
discriminator_loss*a	   ��+O?   ��+O?      �?!   ��+O?) �֣�\�>2IcD���L?k�1^�sO?�������:              �?        
s
generator_loss*a	   �w)C�   �w)C�      �?!   �w)C�) ��G��>2�T���C��!�A��������:              �?        H
"�       b�D�	h|�ש��A�*�
w
discriminator_loss*a	   �݅U?   �݅U?      �?!   �݅U?) Q���>2<DKc��T?ܗ�SsW?�������:              �?        
s
generator_loss*a	   ���F�   ���F�      �?!   ���F�) ��]�>2
����G�a�$��{E��������:              �?        ��5�       b�D�	��ש��A�*�
w
discriminator_loss*a	    ��R?    ��R?      �?!    ��R?)@^'�>2nK���LQ?�lDZrS?�������:              �?        
s
generator_loss*a	   ���@�   ���@�      �?!   ���@�) d����>2�!�A����#@��������:              �?        �3�L�       b�D�	�ש��A�*�
w
discriminator_loss*a	    q�R?    q�R?      �?!    q�R?) ^��i�>2nK���LQ?�lDZrS?�������:              �?        
s
generator_loss*a	   ��,@�   ��,@�      �?!   ��,@�) A?ȫY�>2�!�A����#@��������:              �?        ��:o�       b�D�	���ש��A�*�
w
discriminator_loss*a	    �O?    �O?      �?!    �O?)   Bz�>2IcD���L?k�1^�sO?�������:              �?        
s
generator_loss*a	   @mnA�   @mnA�      �?!   @mnA�) �I���>2�!�A����#@��������:              �?        _��W�       b�D�	9��ש��A�*�
w
discriminator_loss*a	   @�0M?   @�0M?      �?!   @�0M?)�DATM��>2IcD���L?k�1^�sO?�������:              �?        
s
generator_loss*a	   �4�@�   �4�@�      �?!   �4�@�) DLnd��>2�!�A����#@��������:              �?        |\�P�       b�D�	̮�ש��A�*�
w
discriminator_loss*a	    �N?    �N?      �?!    �N?) �&i���>2IcD���L?k�1^�sO?�������:              �?        
s
generator_loss*a	   �+X@�   �+X@�      �?!   �+X@�) ��a=��>2�!�A����#@��������:              �?        �]_��       b�D�	���ש��A�*�
w
discriminator_loss*a	    3mK?    3mK?      �?!    3mK?) ��꺁�>2�qU���I?IcD���L?�������:              �?        
s
generator_loss*a	   @?�:�   @?�:�      �?!   @?�:�)����>2��%>��:�uܬ�@8��������:              �?        �'m��       b�D�	{q�ש��A�*�
w
discriminator_loss*a	   �5@E?   �5@E?      �?!   �5@E?) ��Ǝ9�>2�T���C?a�$��{E?�������:              �?        
s
generator_loss*a	    �U8�    �U8�      �?!    �U8�)  H���>2��%>��:�uܬ�@8��������:              �?        D�3|�       b�D�	dI�ש��A�*�
w
discriminator_loss*a	   `��M?   `��M?      �?!   `��M?) �f���>2IcD���L?k�1^�sO?�������:              �?        
s
generator_loss*a	   `b8�   `b8�      �?!   `b8�) '>>2��%>��:�uܬ�@8��������:              �?        ����       b�D�	/�ש��A�*�
w
discriminator_loss*a	   `5-K?   `5-K?      �?!   `5-K?) w���>2�qU���I?IcD���L?�������:              �?        
s
generator_loss*a	   `h<�   `h<�      �?!   `h<�) �z�r7�>2d�\D�X=���%>��:��������:              �?        �D�Y�       b�D�	�
�ש��A�*�
w
discriminator_loss*a	   �l�E?   �l�E?      �?!   �l�E?) į����>2a�$��{E?
����G?�������:              �?        
s
generator_loss*a	    sI7�    sI7�      �?!    sI7�) .�>�>2uܬ�@8���%�V6��������:              �?        Z�Q�       b�D�	���ש��A�*�
w
discriminator_loss*a	   �M?   �M?      �?!   �M?)�P�`�f�>2IcD���L?k�1^�sO?�������:              �?        
s
generator_loss*a	   ��3;�   ��3;�      �?!   ��3;�)�ԣ�k�>2d�\D�X=���%>��:��������:              �?        ?����       b�D�	�A/ة��A�*�
w
discriminator_loss*a	   @�eC?   @�eC?      �?!   @�eC?) 	pnȄ�>2�!�A?�T���C?�������:              �?        
s
generator_loss*a	   @�@3�   @�@3�      �?!   @�@3�) G�+w>2�u�w74���82��������:              �?        6���       b�D�	�3ة��A�*�
w
discriminator_loss*a	   �6ID?   �6ID?      �?!   �6ID?) YS�W��>2�T���C?a�$��{E?�������:              �?        
s
generator_loss*a	   �3{5�   �3{5�      �?!   �3{5�) �u��|>2��%�V6��u�w74��������:              �?        3�m�       b�D�	�7ة��A�*�
w
discriminator_loss*a	    �YH?    �YH?      �?!    �YH?)  V�O��>2
����G?�qU���I?�������:              �?        
s
generator_loss*a	   @q�0�   @q�0�      �?!   @q�0�) �,Lq>2��bȬ�0���VlQ.��������:              �?        ���       b�D�	�;ة��A�*�
w
discriminator_loss*a	   ���B?   ���B?      �?!   ���B?) $^S�k�>2�!�A?�T���C?�������:              �?        
s
generator_loss*a	   @i�9�   @i�9�      �?!   @i�9�)�,�����>2��%>��:�uܬ�@8��������:              �?        ��ş�       b�D�	K�>ة��A�*�
w
discriminator_loss*a	   ���_?   ���_?      �?!   ���_?) B߉�{�>2E��{��^?�l�P�`?�������:              �?        
s
generator_loss*a	   `�N0�   `�N0�      �?!   `�N0�)@��p>2��bȬ�0���VlQ.��������:              �?        �~�/�       b�D�	WBCة��A�*�
w
discriminator_loss*a	    ?�7?    ?�7?      �?!    ?�7?) �3�O�>2��%�V6?uܬ�@8?�������:              �?        
s
generator_loss*a	   @�0(�   @�0(�      �?!   @�0(�)� ��	Ib>2I�I�)�(�+A�F�&��������:              �?        ����       b�D�	�QGة��A�*�
w
discriminator_loss*a	   `}�??   `}�??      �?!   `}�??) 7IzJ��>2d�\D�X=?���#@?�������:              �?        
s
generator_loss*a	   ��.6�   ��.6�      �?!   ��.6�) �O��~>2uܬ�@8���%�V6��������:              �?        ���3�       b�D�	�oKة��A�*�
w
discriminator_loss*a	   ��2D?   ��2D?      �?!   ��2D?)@����>2�T���C?a�$��{E?�������:              �?        
s
generator_loss*a	   @�o'�   @�o'�      �?!   @�o'�)��`I*a>2I�I�)�(�+A�F�&��������:              �?        ����       b�D�	%��ة��A�*�
w
discriminator_loss*a	   �}@?   �}@?      �?!   �}@?) Q���>2���#@?�!�A?�������:              �?        
s
generator_loss*a	   ��6�   ��6�      �?!   ��6�) $tew�~>2uܬ�@8���%�V6��������:              �?        ���F�       b�D�	��ة��A�*�
w
discriminator_loss*a	   �2�7?   �2�7?      �?!   �2�7?)�|��`ف>2��%�V6?uܬ�@8?�������:              �?        
s
generator_loss*a	    �+�    �+�      �?!    �+�)  O�D�f>2�7Kaa+�I�I�)�(��������:              �?        Q���       b�D�	{w�ة��A�*�
w
discriminator_loss*a	   ���:?   ���:?      �?!   ���:?)���6��>2uܬ�@8?��%>��:?�������:              �?        
s
generator_loss*a	   ���(�   ���(�      �?!   ���(�) !냛]c>2�7Kaa+�I�I�)�(��������:              �?        ���       b�D�	��ة��A�*�
w
discriminator_loss*a	   �	�@?   �	�@?      �?!   �	�@?)@F��g�>2���#@?�!�A?�������:              �?        
s
generator_loss*a	    	2�    	2�      �?!    	2�)@4q�Ft>2��82���bȬ�0��������:              �?        \����       b�D�	x��ة��A�*�
w
discriminator_loss*a	   @2>5?   @2>5?      �?!   @2>5?) ��54|>2�u�w74?��%�V6?�������:              �?        
s
generator_loss*a	   �(�   �(�      �?!   �(�)���
b>2I�I�)�(�+A�F�&��������:              �?        w��m�       b�D�	N��ة��A�*�
w
discriminator_loss*a	   @͚H?   @͚H?      �?!   @͚H?)�|�� �>2
����G?�qU���I?�������:              �?        
s
generator_loss*a	    ��.�    ��.�      �?!    ��.�)  s�m>2��bȬ�0���VlQ.��������:              �?        :�!��       b�D�	Ȃ�ة��A�*�
w
discriminator_loss*a	   �Y@?   �Y@?      �?!   �Y@?) �$��"�>2d�\D�X=?���#@?�������:              �?        
s
generator_loss*a	   �g>+�   �g>+�      �?!   �g>+�) �^Z�1g>2�7Kaa+�I�I�)�(��������:              �?        6���       b�D�	�b�ة��A�*�
w
discriminator_loss*a	   ���??   ���??      �?!   ���??) ݖ3|я>2d�\D�X=?���#@?�������:              �?        
s
generator_loss*a	   �2�*�   �2�*�      �?!   �2�*�) ��k�f>2�7Kaa+�I�I�)�(��������:              �?        �-�       b�D�	T%�ة��A�*�
w
discriminator_loss*a	   ��5?   ��5?      �?!   ��5?) WUc�}>2�u�w74?��%�V6?�������:              �?        
s
generator_loss*a	   ��9'�   ��9'�      �?!   ��9'�) b�f�`>2I�I�)�(�+A�F�&��������:              �?        F���       b�D�	��ة��A�*�
w
discriminator_loss*a	    �>4?    �>4?      �?!    �>4?) ���y>2�u�w74?��%�V6?�������:              �?        
s
generator_loss*a	   ���0�   ���0�      �?!   ���0�) d�G�q>2��82���bȬ�0��������:              �?        �OL��       b�D�	%$�ة��A�*�
w
discriminator_loss*a	    �:?    �:?      �?!    �:?) D,��>2uܬ�@8?��%>��:?�������:              �?        
s
generator_loss*a	   ���0�   ���0�      �?!   ���0�) dNٰq>2��82���bȬ�0��������:              �?        4�z�       b�D�	�%�ة��A�*�
w
discriminator_loss*a	    8�0?    8�0?      �?!    8�0?)  �+��q>2��bȬ�0?��82?�������:              �?        
s
generator_loss*a	   �VQ$�   �VQ$�      �?!   �VQ$�) Y.`��Y>2U�4@@�$��[^:��"��������:              �?        �ѱ�       b�D�	��ة��A�*�
w
discriminator_loss*a	   ��8?   ��8?      �?!   ��8?)�(�/%�>2��%�V6?uܬ�@8?�������:              �?        
s
generator_loss*a	    )'#�    )'#�      �?!    )'#�) I8a�V>2U�4@@�$��[^:��"��������:              �?        x+��       b�D�	��ة��A�*�
w
discriminator_loss*a	    Ֆ@?    Ֆ@?      �?!    Ֆ@?) ���73�>2���#@?�!�A?�������:              �?        
s
generator_loss*a	   @t�'�   @t�'�      �?!   @t�'�)�PB�ғa>2I�I�)�(�+A�F�&��������:              �?        In$��       b�D�	�)٩��A�*�
w
discriminator_loss*a	    �:?    �:?      �?!    �:?) �P�A�>2uܬ�@8?��%>��:?�������:              �?        
s
generator_loss*a	    X#�    X#�      �?!    X#�)@0iQcW>2U�4@@�$��[^:��"��������:              �?        �@��       b�D�	s٩��A�*�
w
discriminator_loss*a	   @�,?   @�,?      �?!   @�,?)�P@aI�h>2�7Kaa+?��VlQ.?�������:              �?        
s
generator_loss*a	   @=�'�   @=�'�      �?!   @=�'�)�<�3�Ya>2I�I�)�(�+A�F�&��������:              �?        �y�A�       b�D�	��S٩��A�*�
w
discriminator_loss*a	   ��}=?   ��}=?      �?!   ��}=?) b���-�>2d�\D�X=?���#@?�������:              �?        
s
generator_loss*a	   @�%�   @�%�      �?!   @�%�) A�J�]>2+A�F�&�U�4@@�$��������:              �?        Ox}|�       b�D�	`VX٩��A�*�
w
discriminator_loss*a	   `��.?   `��.?      �?!   `��.?) �Y�S�m>2��VlQ.?��bȬ�0?�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) ��+�I>2ji6�9���.���������:              �?        ���       b�D�	�h]٩��A�*�
w
discriminator_loss*a	    �f8?    �f8?      �?!    �f8?) �t�W��>2uܬ�@8?��%>��:?�������:              �?        
s
generator_loss*a	   @h0!�   @h0!�      �?!   @h0!�) A'�owR>2�[^:��"��S�F !��������:              �?        >���       b�D�	��a٩��A�*�
w
discriminator_loss*a	    �	8?    �	8?      �?!    �	8?) �RD�>2��%�V6?uܬ�@8?�������:              �?        
s
generator_loss*a	    �!�    �!�      �?!    �!�)@�z�uDR>2�[^:��"��S�F !��������:              �?        �{��       b�D�	��e٩��A�*�
w
discriminator_loss*a	   @(0?   @(0?      �?!   @(0?) A�Q[p>2��VlQ.?��bȬ�0?�������:              �?        
s
generator_loss*a	   �]�*�   �]�*�      �?!   �]�*�) 2	���f>2�7Kaa+�I�I�)�(��������:              �?        �˯$�       b�D�	�'j٩��A�*�
w
discriminator_loss*a	   ��0?   ��0?      �?!   ��0?)@6�:p>2��VlQ.?��bȬ�0?�������:              �?        
s
generator_loss*a	   ���!�   ���!�      �?!   ���!�) )u��S>2�[^:��"��S�F !��������:              �?        �A���       b�D�	��n٩��A�*�
w
discriminator_loss*a	   �^1?   �^1?      �?!   �^1?) ��b�r>2��bȬ�0?��82?�������:              �?        
s
generator_loss*a	   �� �   �� �      �?!   �� �)@n���9Q>2�S�F !�ji6�9���������:              �?        (Z�       b�D�	Ür٩��A�*�
w
discriminator_loss*a	   @�A0?   @�A0?      �?!   @�A0?) ٭���p>2��VlQ.?��bȬ�0?�������:              �?        
s
generator_loss*a	    I��    I��      �?!    I��) ��Nx�K>2ji6�9���.���������:              �?        ����       b�D�	�n�٩��A�*�
w
discriminator_loss*a	   �t5?   �t5?      �?!   �t5?) �I���|>2�u�w74?��%�V6?�������:              �?        
s
generator_loss*a	   `�x$�   `�x$�      �?!   `�x$�)@�L�0Z>2U�4@@�$��[^:��"��������:              �?        ��R��       b�D�	��٩��A�*�
w
discriminator_loss*a	   @��2?   @��2?      �?!   @��2?) ����v>2��82?�u�w74?�������:              �?        
s
generator_loss*a	   ���%�   ���%�      �?!   ���%�) ��ᝀ]>2+A�F�&�U�4@@�$��������:              �?        � ��       b�D�	g��٩��A�*�
w
discriminator_loss*a	   �P0$?   �P0$?      �?!   �P0$?)@��\yY>2�[^:��"?U�4@@�$?�������:              �?        
s
generator_loss*a	   �ɢ�   �ɢ�      �?!   �ɢ�)���#>uA>2��ڋ��vV�R9��������:              �?        �q���       b�D�	���٩��A�*�
w
discriminator_loss*a	   `+�1?   `+�1?      �?!   `+�1?)@�-��s>2��bȬ�0?��82?�������:              �?        
s
generator_loss*a	   ��� �   ��� �      �?!   ��� �) �ܝ�Q>2�S�F !�ji6�9���������:              �?        ����       b�D�	U��٩��A�*�
w
discriminator_loss*a	    %�(?    %�(?      �?!    %�(?) O��'c>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��)  Od��D>2�.����ڋ��������:              �?        ����       b�D�	1��٩��A�*�
w
discriminator_loss*a	   �xk4?   �xk4?      �?!   �xk4?)@,E�z>2�u�w74?��%�V6?�������:              �?        
s
generator_loss*a	   @��!�   @��!�      �?!   @��!�) ��r�S>2�[^:��"��S�F !��������:              �?        ����       b�D�	Q�٩��A�*�
w
discriminator_loss*a	   @6)?   @6)?      �?!   @6)?)��4���c>2I�I�)�(?�7Kaa+?�������:              �?        
s
generator_loss*a	   �m/�   �m/�      �?!   �m/�) �s*dN>2�S�F !�ji6�9���������:              �?        �S*��       b�D�	>�٩��A�*�
w
discriminator_loss*a	   �$/.?   �$/.?      �?!   �$/.?) �a�xl>2��VlQ.?��bȬ�0?�������:              �?        
s
generator_loss*a	    �h�    �h�      �?!    �h�) �lX�E>2�.����ڋ��������:              �?        �	���       b�D�	�Cک��A�*�
w
discriminator_loss*a	   ��
)?   ��
)?      �?!   ��
)?)���M%�c>2I�I�)�(?�7Kaa+?�������:              �?        
s
generator_loss*a	   �)�   �)�      �?!   �)�)�xf(�F>2�.����ڋ��������:              �?        z�=z�       b�D�	SBGک��A�*�
w
discriminator_loss*a	    ��9?    ��9?      �?!    ��9?)  ���>2uܬ�@8?��%>��:?�������:              �?        
s
generator_loss*a	   `��$�   `��$�      �?!   `��$�)@)�9
[>2+A�F�&�U�4@@�$��������:              �?        �]��       b�D�	��Kک��A�*�
w
discriminator_loss*a	   `��-?   `��-?      �?!   `��-?) O�egl>2�7Kaa+?��VlQ.?�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��) �73�F>2�.����ڋ��������:              �?        ���;�       b�D�	��Oک��A�*�
w
discriminator_loss*a	    ԩ)?    ԩ)?      �?!    ԩ)?) (���d>2I�I�)�(?�7Kaa+?�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) A��e�;>2�T7����5�i}1��������:              �?        L,���       b�D�	<Tک��A�*�
w
discriminator_loss*a	   ���.?   ���.?      �?!   ���.?) �c��m>2��VlQ.?��bȬ�0?�������:              �?        
s
generator_loss*a	   �v2&�   �v2&�      �?!   �v2&�) Y��e�^>2+A�F�&�U�4@@�$��������:              �?        C����       b�D�	?&Xک��A�*�
w
discriminator_loss*a	   `��#?   `��#?      �?!   `��#?)@:1�W>2�[^:��"?U�4@@�$?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) �r�=>2�vV�R9��T7����������:              �?        �����       b�D�	P$\ک��A�*�
w
discriminator_loss*a	   �c�0?   �c�0?      �?!   �c�0?)@R��@q>2��VlQ.?��bȬ�0?�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)��zX��G>2�.����ڋ��������:              �?        I�m&�       b�D�	Jn`ک��A�*�
w
discriminator_loss*a	   ���%?   ���%?      �?!   ���%?) ����|]>2U�4@@�$?+A�F�&?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@z�H�8>2�T7����5�i}1��������:              �?        w���       b�D�	���ک��A�*�
w
discriminator_loss*a	   ���&?   ���&?      �?!   ���&?)��WX`>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	   �8�   �8�      �?!   �8�) �E���J>2ji6�9���.���������:              �?        %���       b�D�	O��ک��A�*�
w
discriminator_loss*a	   @�'?   @�'?      �?!   @�'?)�����a>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	   �K� �   �K� �      �?!   �K� �)@�K`��Q>2�S�F !�ji6�9���������:              �?        ��Z�       b�D�	d��ک��A�*�
w
discriminator_loss*a	   ��$4?   ��$4?      �?!   ��$4?) q��[y>2�u�w74?��%�V6?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) 	�X8;>2�T7����5�i}1��������:              �?        �%q��       b�D�	(��ک��A�*�
w
discriminator_loss*a	   ��8+?   ��8+?      �?!   ��8+?)�8��(g>2I�I�)�(?�7Kaa+?�������:              �?        
s
generator_loss*a	    �;�    �;�      �?!    �;�) ��	Y.<>2�vV�R9��T7����������:              �?        ed��       b�D�	���ک��A�*�
w
discriminator_loss*a	   �D+?   �D+?      �?!   �D+?) 	t�q;g>2I�I�)�(?�7Kaa+?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) �F>2�.����ڋ��������:              �?        �b&3�       b�D�	 ��ک��A�*�
w
discriminator_loss*a	   @`4?   @`4?      �?!   @`4?) �
�a�y>2�u�w74?��%�V6?�������:              �?        
s
generator_loss*a	   �O�#�   �O�#�      �?!   �O�#�) ��|�W>2U�4@@�$��[^:��"��������:              �?        Ѥz�       b�D�	���ک��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)��sLM�O>2ji6�9�?�S�F !?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ]o�9�/>2��d�r�x?�x���������:              �?        �����       b�D�	n��ک��A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)��r�+H>2��ڋ?�.�?�������:              �?        
s
generator_loss*a	   �x��   �x��      �?!   �x��) �;���8>2�T7����5�i}1��������:              �?        MkA��       b�D�	V_E۩��A�*�
w
discriminator_loss*a	   ��s?   ��s?      �?!   ��s?) �ț��L>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) �6�'>2>h�'��f�ʜ�7
��������:              �?        �Þ�       b�D�	T>I۩��A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) @,D�96>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   @E
�   @E
�      �?!   @E
�)��)�<%>2f�ʜ�7
�������������:              �?        mKR��       b�D�	":M۩��A�*�
w
discriminator_loss*a	   ���'?   ���'?      �?!   ���'?)�(���Za>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	    �p�    �p�      �?!    �p�) (^��L>2ji6�9���.���������:              �?        �=�;�       b�D�	�5Q۩��A�*�
w
discriminator_loss*a	   `��&?   `��&?      �?!   `��&?) ե0:`>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	   @ ��   @ ��      �?!   @ ��)� L��AC>2��ڋ��vV�R9��������:              �?        �s���       b�D�	�U۩��A�*�
w
discriminator_loss*a	   ��'?   ��'?      �?!   ��'?)���
�Ja>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	   �ћ�   �ћ�      �?!   �ћ�)��R�beK>2ji6�9���.���������:              �?        �E���       b�D�	P�X۩��A�*�
w
discriminator_loss*a	   @�$%?   @�$%?      �?!   @�$%?) �;���[>2U�4@@�$?+A�F�&?�������:              �?        
s
generator_loss*a	   �	��   �	��      �?!   �	��) �U��T1>2��d�r�x?�x���������:              �?        ����       b�D�	��\۩��A�*�
w
discriminator_loss*a	   @��!?   @��!?      �?!   @��!?) an�+T>2�S�F !?�[^:��"?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  �L��B>2��ڋ��vV�R9��������:              �?        2�;6�       b�D�	��`۩��A�*�
w
discriminator_loss*a	    [�?    [�?      �?!    [�?) ~'���D>2��ڋ?�.�?�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) y�"�7>2�T7����5�i}1��������:              �?        ����       b�D�	0s�۩��A�*�
w
discriminator_loss*a	   `&6-?   `&6-?      �?!   `&6-?) �/��j>2�7Kaa+?��VlQ.?�������:              �?        
s
generator_loss*a	    6r�    6r�      �?!    6r�) @6�o�7>2�T7����5�i}1��������:              �?        �'z�       b�D�	�d�۩��A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)��S��A>2�vV�R9?��ڋ?�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�@�&/>2x?�x��>h�'���������:              �?        �̟��       b�D�	�Y�۩��A�*�
w
discriminator_loss*a	   ��_(?   ��_(?      �?!   ��_(?) ��㦐b>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ��1(>2>h�'��f�ʜ�7
��������:              �?        OE>�       b�D�	���۩��A�*�
w
discriminator_loss*a	   �=�)?   �=�)?      �?!   �=�)?)�(wa�d>2I�I�)�(?�7Kaa+?�������:              �?        
s
generator_loss*a	   �T�   �T�      �?!   �T�) �n�!2>2��d�r�x?�x���������:              �?        ����       b�D�	͡�۩��A�*�
w
discriminator_loss*a	   ��,?   ��,?      �?!   ��,?) XDc|j>2�7Kaa+?��VlQ.?�������:              �?        
s
generator_loss*a	   @�d�   @�d�      �?!   @�d�)���-Vs'>2>h�'��f�ʜ�7
��������:              �?        �`�       b�D�	���۩��A�*�
w
discriminator_loss*a	    a?    a?      �?!    a?)@�i�0>2x?�x�?��d�r?�������:              �?        
s
generator_loss*a	   @w��   @w��      �?!   @w��) �Ȩ?l>2�����6�]����������:              �?        ��W��       b�D�	͹�۩��A�*�
w
discriminator_loss*a	   ��#?   ��#?      �?!   ��#?) 	��m�V>2�[^:��"?U�4@@�$?�������:              �?        
s
generator_loss*a	   �Q�   �Q�      �?!   �Q�) 5ܫ*>2x?�x��>h�'���������:              �?        �K;��       b�D�	���۩��A�*�
w
discriminator_loss*a	   �%m?   �%m?      �?!   �%m?)@z�o?>2�T7��?�vV�R9?�������:              �?        
s
generator_loss*a	    I�    I�      �?!    I�)@��lS�0>2��d�r�x?�x���������:              �?        Ń|,�       b�D�	E�Zܩ��A�*�
w
discriminator_loss*a	    k�)?    k�)?      �?!    k�)?) ȅ�;\d>2I�I�)�(?�7Kaa+?�������:              �?        
s
generator_loss*a	   ��/�   ��/�      �?!   ��/�) d"q��4>2�5�i}1���d�r��������:              �?        +�OZ�       b�D�	��^ܩ��A�*�
w
discriminator_loss*a	   @�'?   @�'?      �?!   @�'?)��e�+�a>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	    6C�    6C�      �?!    6C�) �E_�*>2x?�x��>h�'���������:              �?        �`+�       b�D�	�bܩ��A�*�
w
discriminator_loss*a	   ��(.?   ��(.?      �?!   ��(.?) _ƾ/ll>2��VlQ.?��bȬ�0?�������:              �?        
s
generator_loss*a	   �t��   �t��      �?!   �t��) "�<�M>2ji6�9���.���������:              �?        ����       b�D�	a�fܩ��A�*�
w
discriminator_loss*a	   `�� ?   `�� ?      �?!   `�� ?)@t�qQ>2ji6�9�?�S�F !?�������:              �?        
s
generator_loss*a	   �K�   �K�      �?!   �K�)���IGG>2�.����ڋ��������:              �?        �a��       b�D�	��jܩ��A�*�
w
discriminator_loss*a	   ���!?   ���!?      �?!   ���!?) �(��(T>2�S�F !?�[^:��"?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) ��31>2��d�r�x?�x���������:              �?        s"���       b�D�	ݤnܩ��A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@�m�E2>2x?�x�?��d�r?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) �.i=s#>2f�ʜ�7
�������������:              �?        �h�       b�D�	ګrܩ��A�*�
w
discriminator_loss*a	    � ?    � ?      �?!    � ?)@x)^Q>2ji6�9�?�S�F !?�������:              �?        
s
generator_loss*a	   �6~�   �6~�      �?!   �6~�) ?�T`?A>2��ڋ��vV�R9��������:              �?        \���       b�D�	�vܩ��A�*�
w
discriminator_loss*a	   ��x?   ��x?      �?!   ��x?) ����E>2��ڋ?�.�?�������:              �?        
s
generator_loss*a	   �p��   �p��      �?!   �p��)�D���+*>2x?�x��>h�'���������:              �?        m�"��       b�D�	��ܩ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) !tk:;;>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ��؜�+>2x?�x��>h�'���������:              �?        �a(�       b�D�	���ܩ��A�*�
w
discriminator_loss*a	   �Y9?   �Y9?      �?!   �Y9?)@
�8��4>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   ��	�   ��	�      �?!   ��	�) �u��� >2�����6�]����������:              �?        �N�       b�D�	���ܩ��A�*�
w
discriminator_loss*a	   `�?   `�?      �?!   `�?)@^9�jF:>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	   @\��   @\��      �?!   @\��) �gp>21��a˲���[���������:              �?        E��       b�D�	U��ܩ��A�*�
w
discriminator_loss*a	   @�e?   @�e?      �?!   @�e?)�Lw?��B>2�vV�R9?��ڋ?�������:              �?        
s
generator_loss*a	   ��4��   ��4��      �?!   ��4��)��T�Yn>2�FF�G �>�?�s����������:              �?        �~�8�       b�D�	l��ܩ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) A=�KA>2�vV�R9?��ڋ?�������:              �?        
s
generator_loss*a	   �f�   �f�      �?!   �f�) �ioe�>2��[���FF�G ��������:              �?        =k���       b�D�	�ݩ��A�*�
w
discriminator_loss*a	    4�?    4�?      �?!    4�?) �:��+>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   ��y�   ��y�      �?!   ��y�)@=�>2��[���FF�G ��������:              �?        ��MI�       b�D�	�Kݩ��A�*�
w
discriminator_loss*a	   ��h"?   ��h"?      �?!   ��h"?) D@��-U>2�S�F !?�[^:��"?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �I�k�I>2ji6�9���.���������:              �?        �J���       b�D�	�fݩ��A�*�
w
discriminator_loss*a	   ��A%?   ��A%?      �?!   ��A%?) ���e=\>2U�4@@�$?+A�F�&?�������:              �?        
s
generator_loss*a	   ও�   ও�      �?!   ও�) :���(/>2x?�x��>h�'���������:              �?        tO�`�       b�D�	:ʋݩ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) ��
�=>2�T7��?�vV�R9?�������:              �?        
s
generator_loss*a	   `fi�   `fi�      �?!   `fi�) �S� !>2�����6�]����������:              �?        ��4�       b�D�	E��ݩ��A�*�
w
discriminator_loss*a	    �9#?    �9#?      �?!    �9#?)@���W>2�[^:��"?U�4@@�$?�������:              �?        
s
generator_loss*a	   ��_�   ��_�      �?!   ��_�) �J�Q�*>2x?�x��>h�'���������:              �?        A�tr�       b�D�	���ݩ��A�*�
w
discriminator_loss*a	   `5�?   `5�?      �?!   `5�?) ia�xD>2��ڋ?�.�?�������:              �?        
s
generator_loss*a	   @��   @��      �?!   @��) )F��>21��a˲���[���������:              �?        �N��       b�D�	���ݩ��A�*�
w
discriminator_loss*a	   @�?   @�?      �?!   @�?)�T#I@>2�T7��?�vV�R9?�������:              �?        
s
generator_loss*a	   �%	�   �%	�      �?!   �%	�) 2S$)�#>2f�ʜ�7
�������������:              �?        .���       b�D�	Qʛݩ��A�*�
w
discriminator_loss*a	    ��0?    ��0?      �?!    ��0?)@����q>2��bȬ�0?��82?�������:              �?        
s
generator_loss*a	   �]�   �]�      �?!   �]�)@�k���6>2�5�i}1���d�r��������:              �?        �����       b�D�	�ןݩ��A�*�
w
discriminator_loss*a	    j�%?    j�%?      �?!    j�%?) @�;��]>2U�4@@�$?+A�F�&?�������:              �?        
s
generator_loss*a	   �Sy�   �Sy�      �?!   �Sy�) an�i3>2�5�i}1���d�r��������:              �?        �����       b�D�	(�ݩ��A�*�
w
discriminator_loss*a	   �ϸ?   �ϸ?      �?!   �ϸ?) o]�:>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��)@l�`@>>26�]���1��a˲��������:              �?        ݼd��       b�D�	��ݩ��A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)��bJ�1+>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)@F=���>21��a˲���[���������:              �?        ~!&.�       b�D�	%.ީ��A�*�
w
discriminator_loss*a	   `��?   `��?      �?!   `��?) ��`�!>2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	   @_�   @_�      �?!   @_�) 	W@�m>21��a˲���[���������:              �?        ���       b�D�	2ީ��A�*�
w
discriminator_loss*a	   `�F?   `�F?      �?!   `�F?) {�R��*>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	    � �    � �      �?!    � �) ����>2�FF�G �>�?�s����������:              �?        �����       b�D�	��5ީ��A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) ��Hv�K>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	   �� ��   �� ��      �?!   �� ��)�q�e!>2�FF�G �>�?�s����������:              �?        ��aV�       b�D�	��9ީ��A�*�
w
discriminator_loss*a	   �_/	?   �_/	?      �?!   �_/	?) ��K�#>2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ��F>26�]���1��a˲��������:              �?        8�kt�       b�D�	�=ީ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �z��D>2��ڋ?�.�?�������:              �?        
s
generator_loss*a	   `��   `��      �?!   `��) Eb�a9,>2x?�x��>h�'���������:              �?        �D�]�       b�D�	��Aީ��A�*�
w
discriminator_loss*a	   �F� ?   �F� ?      �?!   �F� ?)@���Q>2ji6�9�?�S�F !?�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�w�
T >2�����6�]����������:              �?        �a
�       b�D�	�AFީ��A�*�
w
discriminator_loss*a	   �o�?   �o�?      �?!   �o�?) �Ԥa/>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   ��+�   ��+�      �?!   ��+�) ��!Dm>26�]���1��a˲��������:              �?        ��o$�       b�D�	M`Jީ��A�*�
w
discriminator_loss*a	   @^x?   @^x?      �?!   @^x?)����iTI>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	   @]C �   @]C �      �?!   @]C �) yև>2�FF�G �>�?�s����������:              �?        bS��       b�D�	۵�ީ��A�*�
w
discriminator_loss*a	    ��%?    ��%?      �?!    ��%?) �kG&�]>2U�4@@�$?+A�F�&?�������:              �?        
s
generator_loss*a	   @7�   @7�      �?!   @7�)�dϵ�.>2x?�x��>h�'���������:              �?        �"{��       b�D�	���ީ��A�*�
w
discriminator_loss*a	   @�!?   @�!?      �?!   @�!?) ��4	qS>2�S�F !?�[^:��"?�������:              �?        
s
generator_loss*a	   �"��   �"��      �?!   �"��) d
`�~>26�]���1��a˲��������:              �?        ��g�       b�D�	ӳ�ީ��A�*�
w
discriminator_loss*a	   @�w?   @�w?      �?!   @�w?) y!Y�.:>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@Ɯ)�L>2��[���FF�G ��������:              �?        	O'�       b�D�	ɖ�ީ��A�*�
w
discriminator_loss*a	   �+�?   �+�?      �?!   �+�?) D6�j%4>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) ܊�(�>2>�?�s���O�ʗ����������:              �?        4���       b�D�	ė�ީ��A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)�4��:�)>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   @�� �   @�� �      �?!   @�� �) )S��Q>2��[���FF�G ��������:              �?        )q���       b�D�	���ީ��A�*�
w
discriminator_loss*a	   @�*?   @�*?      �?!   @�*?)��i(�*>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   ࿣�   ࿣�      �?!   ࿣�)@ q���>21��a˲���[���������:              �?        s��       b�D�	�z�ީ��A�*�
w
discriminator_loss*a	   @�!?   @�!?      �?!   @�!?)�x�\6IN>2ji6�9�?�S�F !?�������:              �?        
s
generator_loss*a	    O?�    O?�      �?!    O?�) �Q�&�.>2x?�x��>h�'���������:              �?        �X���       b�D�	Cc�ީ��A�*�
w
discriminator_loss*a	   �o0?   �o0?      �?!   �o0?)@@NZ��4>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   �	�   �	�      �?!   �	�)@`���U>21��a˲���[���������:              �?        ���       b�D�	�̃ߩ��A�*�
w
discriminator_loss*a	    )�?    )�?      �?!    )�?) )�
=>2�T7��?�vV�R9?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) �[�/>2x?�x��>h�'���������:              �?        ��bV�       b�D�	�هߩ��A�*�
w
discriminator_loss*a	   @_�?   @_�?      �?!   @_�?)����@>2�T7��?�vV�R9?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ����>26�]���1��a˲��������:              �?        s��=�       b�D�	ZӋߩ��A�*�
w
discriminator_loss*a	   ��s?   ��s?      �?!   ��s?) �)I�+>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)@r���=>26�]���1��a˲��������:              �?        �DJ7�       b�D�	��ߩ��A�*�
w
discriminator_loss*a	   @G�?   @G�?      �?!   @G�?) IQW�5>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   `�	�   `�	�      �?!   `�	�)@8��[>2�����6�]����������:              �?        
o��       b�D�	D�ߩ��A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)���L#>2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	   �(���   �(���      �?!   �(���) ɟ��]�=2I��P=��pz�w�7���������:              �?        7  �       b�D�	�ؗߩ��A�*�
w
discriminator_loss*a	   ���#?   ���#?      �?!   ���#?) D<���X>2�[^:��"?U�4@@�$?�������:              �?        
s
generator_loss*a	   ���
�   ���
�      �?!   ���
�) r}{�&>2>h�'��f�ʜ�7
��������:              �?        =��       b�D�	S�ߩ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �U7t�M>2ji6�9�?�S�F !?�������:              �?        
s
generator_loss*a	   ��7��   ��7��      �?!   ��7��) �X!�
>2>�?�s���O�ʗ����������:              �?        _爗�       b�D�	��ߩ��A�*�
w
discriminator_loss*a	    �?    �?      �?!    �?) �4E�:>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��) ��#">2f�ʜ�7
�������������:              �?        R����       b�D�	6[=���A�*�
w
discriminator_loss*a	    .v?    .v?      �?!    .v?)  �s��">2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��) Y�>2��[���FF�G ��������:              �?        ��2��       b�D�	�A���A�*�
w
discriminator_loss*a	    ;[?    ;[?      �?!    ;[?) >��� )>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)@$-&>d>21��a˲���[���������:              �?        ���L�       b�D�	
�E���A�*�
w
discriminator_loss*a	   ��(0?   ��(0?      �?!   ��(0?) �7��Qp>2��VlQ.?��bȬ�0?�������:              �?        
s
generator_loss*a	   �-��   �-��      �?!   �-��)�h�O�>2O�ʗ�����Zr[v���������:              �?        �r2:�       b�D�	0�I���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) �����5>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   @�N�   @�N�      �?!   @�N�) �KD�>21��a˲���[���������:              �?        �9�2�       b�D�	.�M���A�*�
w
discriminator_loss*a	    ,O?    ,O?      �?!    ,O?) �|�PI>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	   �(4��   �(4��      �?!   �(4��) ��g��=2I��P=��pz�w�7���������:              �?        �w���       b�D�	FR���A�*�
w
discriminator_loss*a	    T?    T?      �?!    T?)@ T���>2��[�?1��a˲?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ����>2>�?�s���O�ʗ����������:              �?        ��N��       b�D�	�-V���A�*�
w
discriminator_loss*a	   ��F?   ��F?      �?!   ��F?) #�>26�]��?����?�������:              �?        
s
generator_loss*a	   @T~��   @T~��      �?!   @T~��)��U�;�>2O�ʗ�����Zr[v���������:              �?        ə�V�       b�D�	h3Z���A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) �i49�>26�]��?����?�������:              �?        
s
generator_loss*a	   @����   @����      �?!   @����)�T�k->2O�ʗ�����Zr[v���������:              �?        �r���       b�D�	 ���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �T�$(>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@ I�y<�=2I��P=��pz�w�7���������:              �?        h?K�       b�D�	Ҏ���A�*�
w
discriminator_loss*a	   ��d?   ��d?      �?!   ��d?) o[J��">2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	   �=��   �=��      �?!   �=��)@Z��S��=2I��P=��pz�w�7���������:              �?        H���       b�D�	�	���A�*�
w
discriminator_loss*a	   �D��>   �D��>      �?!   �D��>) �*K�>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	   �@���   �@���      �?!   �@���)�#��>2��Zr[v��I��P=���������:              �?        ��|��       �N�	�����A*�
w
discriminator_loss*a	   �%	?   �%	?      �?!   �%	?) �:P0>2x?�x�?��d�r?�������:              �?        
s
generator_loss*a	    {���    {���      �?!    {���) ����>2�FF�G �>�?�s����������:              �?        q��       �{�	1�����A*�
w
discriminator_loss*a	   ��v?   ��v?      �?!   ��v?) ��V*,>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	    ˕��    ˕��      �?!    ˕��) �w���>2>�?�s���O�ʗ����������:              �?        �l��       �{�	������A
*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)��𧻳F>2��ڋ?�.�?�������:              �?        
s
generator_loss*a	   @/}��   @/}��      �?!   @/}��)��i���>2O�ʗ�����Zr[v���������:              �?        ��\��       �{�	#�����A*�
w
discriminator_loss*a	    ��'?    ��'?      �?!    ��'?) �����a>2+A�F�&?I�I�)�(?�������:              �?        
s
generator_loss*a	   @����   @����      �?!   @����)��#%��>2��Zr[v��I��P=���������:              �?        TG��       �{�	�����A*�
w
discriminator_loss*a	   `QO?   `QO?      �?!   `QO?) �.}�,>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   �����   �����      �?!   �����)���-z>2>�?�s���O�ʗ����������:              �?        O��%�       �{�	:
����A*�
w
discriminator_loss*a	   �Ϭ?   �Ϭ?      �?!   �Ϭ?)�����C>2�vV�R9?��ڋ?�������:              �?        
s
generator_loss*a	   �6 �   �6 �      �?!   �6 �) ��k�l>2�FF�G �>�?�s����������:              �?        �G���       �{�	�����A*�
w
discriminator_loss*a	   �\�?   �\�?      �?!   �\�?) ���R34>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	    �=��    �=��      �?!    �=��)  w�h�>2�FF�G �>�?�s����������:              �?        yq�       �{�	������A#*�
w
discriminator_loss*a	   `��?   `��?      �?!   `��?) ��ŏ�+>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) !��B>26�]���1��a˲��������:              �?        �����       �{�	�����A(*�
w
discriminator_loss*a	   @�?   @�?      �?!   @�?) q'�>>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  ���.�=2pz�w�7��})�l a��������:              �?        t�h�       �{�	=����A-*�
w
discriminator_loss*a	    �?    �?      �?!    �?) @��I>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   �?���   �?���      �?!   �?���) &�i-�=2I��P=��pz�w�7���������:              �?        B��       �{�	�̒���A2*�
w
discriminator_loss*a	   �x?   �x?      �?!   �x?) 2��ŵ">2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	   @ө��   @ө��      �?!   @ө��)���(T>2�FF�G �>�?�s����������:              �?        L��U�       �{�	������A7*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) �S�>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	   @�a��   @�a��      �?!   @�a��)�@�g:n>2>�?�s���O�ʗ����������:              �?        ��C��       �{�	Ƭ����A<*�
w
discriminator_loss*a	   @Ð?   @Ð?      �?!   @Ð?)�T���PK>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	   �q���   �q���      �?!   �q���)�Xh���>2�FF�G �>�?�s����������:              �?        y��I�       �{�	<̞���AA*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) A�E�>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	   ��/��   ��/��      �?!   ��/��)��й��>2O�ʗ�����Zr[v���������:              �?        V[YF�       �{�	�Ȣ���AF*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) 4<"��!>26�]��?����?�������:              �?        
s
generator_loss*a	   �bD��   �bD��      �?!   �bD��) y�)��=2I��P=��pz�w�7���������:              �?        ����       �{�	����AK*�
w
discriminator_loss*a	    �W?    �W?      �?!    �W?) ��3>26�]��?����?�������:              �?        
s
generator_loss*a	    &5��    &5��      �?!    &5��) lw`��
>2>�?�s���O�ʗ����������:              �?        ����       �{�	p�����AP*�
w
discriminator_loss*a	   `x�?   `x�?      �?!   `x�?)@��~I>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) Br�Z�>2��Zr[v��I��P=���������:              �?        ��Fg�       �{�	U�_���AU*�
w
discriminator_loss*a	   `a?   `a?      �?!   `a?) �T�/�*>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   `��   `��      �?!   `��) �EA.��=2�h���`�8K�ߝ��������:              �?        �j%�       �{�	�c���AZ*�
w
discriminator_loss*a	    ŵ?    ŵ?      �?!    ŵ?) �L�C>2�vV�R9?��ڋ?�������:              �?        
s
generator_loss*a	   @sG��   @sG��      �?!   @sG��) )��=2��Zr[v��I��P=���������:              �?        ]��@�       �{�	gtg���A_*�
w
discriminator_loss*a	   `��?   `��?      �?!   `��?) ?�<��K>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	    H��    H��      �?!    H��)  D�G>2��[���FF�G ��������:              �?        I��       �{�	ak���Ad*�
w
discriminator_loss*a	   `I9	?   `I9	?      �?!   `I9	?) ?4��#>2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	   �{#��   �{#��      �?!   �{#��) ���L>2�FF�G �>�?�s����������:              �?        )��       �{�	do���Ai*�
w
discriminator_loss*a	    �?    �?      �?!    �?) �䀲(>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   �� �   �� �      �?!   �� �)@P�>2��[���FF�G ��������:              �?        �[Vj�       �{�	�Ss���An*�
w
discriminator_loss*a	   � ?   � ?      �?!   � ?)@�%�@�;>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	    ?���    ?���      �?!    ?���) ���T >2��Zr[v��I��P=���������:              �?        �m�       �{�	-\w���As*�
w
discriminator_loss*a	    V�?    V�?      �?!    V�?) �e��)*>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   `y��   `y��      �?!   `y��)@� �<$�=2pz�w�7��})�l a��������:              �?        �����       �{�	.P{���Ax*�
w
discriminator_loss*a	    GF�>    GF�>      �?!    GF�>) �=��
>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	   @.���   @.���      �?!   @.���) ���O��=2I��P=��pz�w�7���������:              �?        y����       �{�	��9���A}*�
w
discriminator_loss*a	   `�	?   `�	?      �?!   `�	?) 1�m��$>2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	   @H���   @H���      �?!   @H���) AV�4��=2I��P=��pz�w�7���������:              �?        ���)�       b�D�	b�=���A�*�
w
discriminator_loss*a	   @�?   @�?      �?!   @�?)�����)>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	    ����    ����      �?!    ����)  �B�O >2��Zr[v��I��P=���������:              �?        l�K��       b�D�	�B���A�*�
w
discriminator_loss*a	   @3��>   @3��>      �?!   @3��>)�"?>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	    ����    ����      �?!    ����)@��^J�=2I��P=��pz�w�7���������:              �?        M��S�       b�D�	b`F���A�*�
w
discriminator_loss*a	   �q�?   �q�?      �?!   �q�?) �����>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)���(o��=2�ߊ4F��h���`��������:              �?        �f���       b�D�	�J���A�*�
w
discriminator_loss*a	    M?    M?      �?!    M?) F�ht">2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	    v��    v��      �?!    v��)@|�!{�=2I��P=��pz�w�7���������:              �?        ,U�
�       b�D�	�O���A�*�
w
discriminator_loss*a	    !!?    !!?      �?!    !!?)@��D��>2��[�?1��a˲?�������:              �?        
s
generator_loss*a	    �M��    �M��      �?!    �M��)  y�>2O�ʗ�����Zr[v���������:              �?        ��.�       b�D�	�S���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)���O��>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	    ����    ����      �?!    ����) ������=2I��P=��pz�w�7���������:              �?        ��+��       b�D�	GW���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �nl�K>2�.�?ji6�9�?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �PY�^>2�FF�G �>�?�s����������:              �?        ����       b�D�	����A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?)  �<Q�!>2����?f�ʜ�7
?�������:              �?        
s
generator_loss*a	   �����   �����      �?!   �����)��p�3�>2�FF�G �>�?�s����������:              �?        (Q�       b�D�	�� ���A�*�
w
discriminator_loss*a	   �U�?   �U�?      �?!   �U�?) �Ñ>(=>2�T7��?�vV�R9?�������:              �?        
s
generator_loss*a	   @O��   @O��      �?!   @O��)�D�z�C#>2f�ʜ�7
�������������:              �?        ����       b�D�	��$���A�*�
w
discriminator_loss*a	    .�?    .�?      �?!    .�?) @ҳ�>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   �"��   �"��      �?!   �"��) ��|��=2�ߊ4F��h���`��������:              �?        �ڭ��       b�D�	�f(���A�*�
w
discriminator_loss*a	   ��\
?   ��\
?      �?!   ��\
?) ��g�%>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   @�j �   @�j �      �?!   @�j �) A����>2��[���FF�G ��������:              �?        �c�       b�D�	m�,���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ����]>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	    $��    $��      �?!    $��) ���`T�=28K�ߝ�a�Ϭ(��������:              �?        ׸6�       b�D�	��0���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@\X�z�>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ih^��=2})�l a��ߊ4F���������:              �?        \�(�       b�D�	��4���A�*�
w
discriminator_loss*a	   �h>?   �h>?      �?!   �h>?)@&����2>2x?�x�?��d�r?�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��) @L;�>21��a˲���[���������:              �?        �q�c�       b�D�	�8���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)��o��>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	   ��U�   ��U�      �?!   ��U�)@Lv��]�=2pz�w�7��})�l a��������:              �?        ����       b�D�	�i���A�*�
w
discriminator_loss*a	   `� ?   `� ?      �?!   `� ?)@�ߨ.>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@�1��+�=2})�l a��ߊ4F���������:              �?        �zi��       b�D�	�����A�*�
w
discriminator_loss*a	   @ ��>   @ ��>      �?!   @ ��>)��P{F7>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   `C�   `C�      �?!   `C�)@�c�6r�=2})�l a��ߊ4F���������:              �?        �y�j�       b�D�	Z]���A�*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) �bC��>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   ��}�   ��}�      �?!   ��}�) 	�^�=2pz�w�7��})�l a��������:              �?        {�p��       b�D�	�����A�*�
w
discriminator_loss*a	   `��?   `��?      �?!   `��?)@���2>2x?�x�?��d�r?�������:              �?        
s
generator_loss*a	   ��d��   ��d��      �?!   ��d��) D��=2I��P=��pz�w�7���������:              �?        ��K��       b�D�	wB"���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@0n���1>2x?�x�?��d�r?�������:              �?        
s
generator_loss*a	    �)��    �)��      �?!    �)��) n���>>2��Zr[v��I��P=���������:              �?        ��Zr�       b�D�	6'���A�*�
w
discriminator_loss*a	   `�m?   `�m?      �?!   `�m?)@�k�i95>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	    g
�    g
�      �?!    g
�) W���=2�ߊ4F��h���`��������:              �?        �;�=�       b�D�	��,���A�*�
w
discriminator_loss*a	   ��E?   ��E?      �?!   ��E?)��;��,>2>h�'�?x?�x�?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)��>u>2O�ʗ�����Zr[v���������:              �?        NT$��       b�D�	�T2���A�*�
w
discriminator_loss*a	   �Y?   �Y?      �?!   �Y?)@�`��6>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   �����   �����      �?!   �����)@�����=2I��P=��pz�w�7���������:              �?        _h��       b�D�	��+���A�*�
w
discriminator_loss*a	   �� ?   �� ?      �?!   �� ?) )�7�">2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   �Ƶ��   �Ƶ��      �?!   �Ƶ��) ����>2O�ʗ�����Zr[v���������:              �?        ��T��       b�D�	�/���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) ��Ș>26�]��?����?�������:              �?        
s
generator_loss*a	   @d3�   @d3�      �?!   @d3�) !��mg�=2�ߊ4F��h���`��������:              �?        ���       b�D�	"$4���A�*�
w
discriminator_loss*a	    �K�>    �K�>      �?!    �K�>) �{N�>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   ��y�   ��y�      �?!   ��y�) 5�����=2�ߊ4F��h���`��������:              �?        HL�e�       b�D�	;f8���A�*�
w
discriminator_loss*a	   �ml�>   �ml�>      �?!   �ml�>)�hH.��>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   @ܒ�   @ܒ�      �?!   @ܒ�)����5�=2�ߊ4F��h���`��������:              �?        ���A�       b�D�	6J=���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@����>>2��[�?1��a˲?�������:              �?        
s
generator_loss*a	   ��c�   ��c�      �?!   ��c�)�,e
7%�=28K�ߝ�a�Ϭ(��������:              �?        �u���       b�D�	��A���A�*�
w
discriminator_loss*a	   ��<
?   ��<
?      �?!   ��<
?) bc���%>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   @O���   @O���      �?!   @O���)�D4k> >2��Zr[v��I��P=���������:              �?        �q���       b�D�	j�E���A�*�
w
discriminator_loss*a	   ��x�>   ��x�>      �?!   ��x�>)��ק��>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �7R`0�=2�uE���⾮��%��������:              �?        �����       b�D�	�FJ���A�*�
w
discriminator_loss*a	   �e�!?   �e�!?      �?!   �e�!?)@zU��-S>2�S�F !?�[^:��"?�������:              �?        
s
generator_loss*a	   �=�   �=�      �?!   �=�) ����:�=2�ߊ4F��h���`��������:              �?        Z�6G�       b�D�	<�5���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@7Q�<>26�]��?����?�������:              �?        
s
generator_loss*a	   �٫��   �٫��      �?!   �٫��) lUz� >2��Zr[v��I��P=���������:              �?        �.� �       b�D�	��:���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) 	�&S>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   `>��   `>��      �?!   `>��) �=]v<�=2�ߊ4F��h���`��������:              �?        =GC�       b�D�	J�?���A�*�
w
discriminator_loss*a	   ��e?   ��e?      �?!   ��e?) ��+/Z>26�]��?����?�������:              �?        
s
generator_loss*a	   @?$�   @?$�      �?!   @?$�)���S�=28K�ߝ�a�Ϭ(��������:              �?        H��       b�D�	�D���A�*�
w
discriminator_loss*a	    � ?    � ?      �?!    � ?) H�3�(>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   �N��   �N��      �?!   �N��)@�4h���=2pz�w�7��})�l a��������:              �?        ���M�       b�D�	��I���A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?)  SH} >26�]��?����?�������:              �?        
s
generator_loss*a	   �3��   �3��      �?!   �3��)@0Ԋ���=2pz�w�7��})�l a��������:              �?        ��~)�       b�D�	,N���A�*�
w
discriminator_loss*a	    ɤ?    ɤ?      �?!    ɤ?) ]be�>2��[�?1��a˲?�������:              �?        
s
generator_loss*a	    '�    '�      �?!    '�)@�i,��=2��(��澢f�����������:              �?        >��(�       b�D�	>:R���A�*�
w
discriminator_loss*a	   `l�>   `l�>      �?!   `l�>) 	�D��>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	   `o��   `o��      �?!   `o��)@J�΀t�=2��Zr[v��I��P=���������:              �?        �&��       b�D�	}UV���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��>@�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �$�|��=2a�Ϭ(���(����������:              �?        �;	%�       b�D�	�W���A�*�
w
discriminator_loss*a	   �3�?   �3�?      �?!   �3�?) �%�(�>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	   �M��   �M��      �?!   �M��) $����=2I��P=��pz�w�7���������:              �?        ����       b�D�	�]���A�*�
w
discriminator_loss*a	   @N�>   @N�>      �?!   @N�>)�@@��>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   @@��   @@��      �?!   @@��) �!�=9�=2I��P=��pz�w�7���������:              �?        �����       b�D�	K�a���A�*�
w
discriminator_loss*a	    AR�>    AR�>      �?!    AR�>) ��� >2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   �6��   �6��      �?!   �6��) Y��<5�=2})�l a��ߊ4F���������:              �?        �� ��       b�D�	�%g���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  x��>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	    bL�    bL�      �?!    bL�) @X%I��=2})�l a��ߊ4F���������:              �?        v���       b�D�	ؓk���A�*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) ���]>>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	   `�4�   `�4�      �?!   `�4�)@ratC��=2})�l a��ߊ4F���������:              �?        ��I��       b�D�	�p���A�*�
w
discriminator_loss*a	    �?    �?      �?!    �?) �J86H>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	   ��$�   ��$�      �?!   ��$�)�t6����=2�h���`�8K�ߝ��������:              �?        �7�       b�D�	�u���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) "K� '>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   `b���   `b���      �?!   `b���)@ڸy^�=2I��P=��pz�w�7���������:              �?        �)k�       b�D�	�z���A�*�
w
discriminator_loss*a	   @�0�>   @�0�>      �?!   @�0�>) A���=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	     8�     8�      �?!     8�)@� 9��=2��(��澢f�����������:              �?        ]|�8�       b�D�	�1����A�*�
w
discriminator_loss*a	   ��B�>   ��B�>      �?!   ��B�>) Du��/�=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   `Q��   `Q��      �?!   `Q��) ����=2a�Ϭ(���(����������:              �?        7JY�       b�D�	?P����A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)�d��T�!>26�]��?����?�������:              �?        
s
generator_loss*a	   �̪��   �̪��      �?!   �̪��) ��u�	>2>�?�s���O�ʗ����������:              �?        ���       b�D�	ߗ���A�*�
w
discriminator_loss*a	   @1��>   @1��>      �?!   @1��>) ��!�1�=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) d�S���=2���%ᾙѩ�-߾�������:              �?        J����       b�D�	v����A�*�
w
discriminator_loss*a	    �p�>    �p�>      �?!    �p�>) ��N��=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   @*�   @*�      �?!   @*�)��4���=2a�Ϭ(���(����������:              �?        T�D�       b�D�	+@����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �V��M�=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ��6��=2a�Ϭ(���(����������:              �?        �Z���       b�D�	�����A�*�
w
discriminator_loss*a	   �/��>   �/��>      �?!   �/��>) �Q�.>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   @�2�   @�2�      �?!   @�2�) y�s|�=2})�l a��ߊ4F���������:              �?        :-*r�       b�D�	�O����A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) A]���5>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   ��,�   ��,�      �?!   ��,�) F�x���=2�h���`�8K�ߝ��������:              �?        ��_�       b�D�	������A�*�
w
discriminator_loss*a	    1��>    1��>      �?!    1��>) ��,@�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   �v۾   �v۾      �?!   �v۾) ��q��=2E��a�Wܾ�iD*L�پ�������:              �?        �З��       b�D�	Z����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) 9T���=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)�|֡��=28K�ߝ�a�Ϭ(��������:              �?        #��       b�D�	�(����A�*�
w
discriminator_loss*a	    ,J�>    ,J�>      �?!    ,J�>) ���x�=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   �)�   �)�      �?!   �)�)�x�k���=2a�Ϭ(���(����������:              �?        ��t��       b�D�	Y����A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �9r��=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   �3��   �3��      �?!   �3��) Io$�R�=2�h���`�8K�ߝ��������:              �?        f#��       b�D�	�w����A�*�
w
discriminator_loss*a	   @�  ?   @�  ?      �?!   @�  ?) �4��A>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   �Ё�   �Ё�      �?!   �Ё�)@���'�=2})�l a��ߊ4F���������:              �?        �D���       b�D�	�_����A�*�
w
discriminator_loss*a	   @e��>   @e��>      �?!   @e��>)�\�ǒz>2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) �I��9�=2�ߊ4F��h���`��������:              �?        ��L��       b�D�	Tj����A�*�
w
discriminator_loss*a	   బ�>   బ�>      �?!   బ�>) �9A�>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@: �s�=2�f�����uE�����������:              �?        �_�C�       b�D�	�P����A�*�
w
discriminator_loss*a	   `u�?   `u�?      �?!   `u�?)@�1�7>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	    0�    0�      �?!    0�) H �Me�=2�ߊ4F��h���`��������:              �?        8�?\�       b�D�	=V����A�*�
w
discriminator_loss*a	     �?     �?      �?!     �?)@�Q%>2��[�?1��a˲?�������:              �?        
s
generator_loss*a	    ����    ����      �?!    ����) @bu�\�=2I��P=��pz�w�7���������:              �?        �2�K�       b�D�	�p����A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �#��>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) �m-DF�=2��(��澢f�����������:              �?        G�       b�D�	I�����A�*�
w
discriminator_loss*a	    �v�>    �v�>      �?!    �v�>) ����>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   @�Ӿ   @�Ӿ      �?!   @�Ӿ) I���J�=2��>M|Kվ��~]�[Ӿ�������:              �?        M�$��       b�D�	�A����A�*�
w
discriminator_loss*a	    �?    �?      �?!    �?)@8\�*>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)@������=2�uE���⾮��%��������:              �?        2C�l�       b�D�	]w����A�*�
w
discriminator_loss*a	   �#1�>   �#1�>      �?!   �#1�>) �G;���=2�f����>��(���>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) ��;��=2���%ᾙѩ�-߾�������:              �?        V�f��       b�D�	!�����A�*�
w
discriminator_loss*a	   `%4�>   `%4�>      �?!   `%4�>) ���ئ
>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)����%��=2a�Ϭ(���(����������:              �?        C
}��       b�D�	N�����A�*�
w
discriminator_loss*a	   �h��>   �h��>      �?!   �h��>) �Z���=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   ��a�   ��a�      �?!   ��a�) �u�!�=28K�ߝ�a�Ϭ(��������:              �?        ��
�       b�D�	 ����A�*�
w
discriminator_loss*a	    ?    ?      �?!    ?)@1Ғ�6>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	   ��5�   ��5�      �?!   ��5�) ����=2pz�w�7��})�l a��������:              �?        ��ɴ�       b�D�	�8����A�*�
w
discriminator_loss*a	   �_��>   �_��>      �?!   �_��>) E[�>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)��~�2�=2�h���`�8K�ߝ��������:              �?        B��?�       b�D�	3]���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@L�]��=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   @d)��   @d)��      �?!   @d)��)�>;��=2�h���`�8K�ߝ��������:              �?        ��p��       b�D�	�����A�*�
w
discriminator_loss*a	   �&p?   �&p?      �?!   �&p?) ��Ni?>2��[�?1��a˲?�������:              �?        
s
generator_loss*a	    (=�    (=�      �?!    (=�) P,1\�=2a�Ϭ(���(����������:              �?        V�D�       b�D�	�����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@|�����=2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	    �۾    �۾      �?!    �۾) ��q��=2E��a�Wܾ�iD*L�پ�������:              �?        ��(�       b�D�	�0#���A�*�
w
discriminator_loss*a	   @j��>   @j��>      �?!   @j��>) �A�mh�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)��!��=28K�ߝ�a�Ϭ(��������:              �?        :����       b�D�	3�(���A�*�
w
discriminator_loss*a	   �ie?   �ie?      �?!   �ie?) ����>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   @�;�   @�;�      �?!   @�;�) 9(bx�=2���%ᾙѩ�-߾�������:              �?        ��X�       b�D�	�$-���A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)���{; �=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   ��H�   ��H�      �?!   ��H�)@��0�=�=2�f�����uE�����������:              �?        �UW��       b�D�	o_1���A�*�
w
discriminator_loss*a	   `�k�>   `�k�>      �?!   `�k�>)@�4�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   ��O�   ��O�      �?!   ��O�) �&���=2��(��澢f�����������:              �?        DU�z�       b�D�	~�5���A�*�
w
discriminator_loss*a	    �C�>    �C�>      �?!    �C�>)@l�<���=2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	    S7�    S7�      �?!    S7�)@�+U:��=2})�l a��ߊ4F���������:              �?         nv�       b�D�	c�P���A�*�
w
discriminator_loss*a	   �u�?   �u�?      �?!   �u�?) ����>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) �!�S�=2a�Ϭ(���(����������:              �?        J���       b�D�	�yU���A�*�
w
discriminator_loss*a	    F�>    F�>      �?!    F�>) �6�#�>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   @��   @��      �?!   @��) ����(�=2���%ᾙѩ�-߾�������:              �?        / ���       b�D�	@+Z���A�*�
w
discriminator_loss*a	   @k'�>   @k'�>      �?!   @k'�>) �����=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   @�Y�   @�Y�      �?!   @�Y�) �@��=2�uE���⾮��%��������:              �?        �n��       b�D�	�*^���A�*�
w
discriminator_loss*a	   @A&�>   @A&�>      �?!   @A&�>)�}��9>2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   ��3�   ��3�      �?!   ��3�)@�{M5h�=2�ߊ4F��h���`��������:              �?        ���       b�D�	�(b���A�*�
w
discriminator_loss*a	   @\��>   @\��>      �?!   @\��>) ��"3}�=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   ��&�   ��&�      �?!   ��&�) !����=2��(��澢f�����������:              �?        Fr��       b�D�	:f���A�*�
w
discriminator_loss*a	   �n�>   �n�>      �?!   �n�>) �#�!�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   `�   `�      �?!   `�)@�2Q�'�=2���%ᾙѩ�-߾�������:              �?        ��	��       b�D�	/j���A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) Qw����=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) 9�_*�=2���%ᾙѩ�-߾�������:              �?        `]2d�       b�D�	�]n���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@$�+"e�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   `��   `��      �?!   `��)@vcp��=2�uE���⾮��%��������:              �?        Hp�r�       b�D�	j�o���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ���>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��)@Mz"�=2��(��澢f�����������:              �?        z��       b�D�	�t���A�*�
w
discriminator_loss*a	   �D4�>   �D4�>      �?!   �D4�>) >l�V�>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   �/��   �/��      �?!   �/��) ��(�=2a�Ϭ(���(����������:              �?        R-�y�       b�D�	��y���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@�v�m��=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   @�E�   @�E�      �?!   @�E�)��S�i�=2a�Ϭ(���(����������:              �?        ��|��       b�D�	k�}���A�*�
w
discriminator_loss*a	   `)��>   `)��>      �?!   `)��>) ��R�>2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   `�۾   `�۾      �?!   `�۾) u"	2�=2E��a�Wܾ�iD*L�پ�������:              �?        �;|b�       b�D�	q�����A�*�
w
discriminator_loss*a	   �|k?   �|k?      �?!   �|k?) ��4�4>2��[�?1��a˲?�������:              �?        
s
generator_loss*a	   �%�޾   �%�޾      �?!   �%�޾) ���d��=2�ѩ�-߾E��a�Wܾ�������:              �?        >��h�       b�D�	�����A�*�
w
discriminator_loss*a	    �z�>    �z�>      �?!    �z�>) �r=#:�=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   @�q�   @�q�      �?!   @�q�) 9�y��=2�uE���⾮��%��������:              �?        ?_�U�       b�D�	G3����A�*�
w
discriminator_loss*a	    ѱ�>    ѱ�>      �?!    ѱ�>) e����=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   �ھ   �ھ      �?!   �ھ) �m�P�=2E��a�Wܾ�iD*L�پ�������:              �?        ��&��       b�D�	�j����A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) H s5�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) �$+�=2a�Ϭ(���(����������:              �?        ,��       b�D�	�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ������=2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) �@��=2�uE���⾮��%��������:              �?        �'�       b�D�	�}����A�*�
w
discriminator_loss*a	    E��>    E��>      �?!    E��>) �DP$��=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   � 7ݾ   � 7ݾ      �?!   � 7ݾ) ��I��=2�ѩ�-߾E��a�Wܾ�������:              �?        ?(�       b�D�	u����A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)�ȵ#�^�=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	    پ    پ      �?!    پ)  j�O��=2�iD*L�پ�_�T�l׾�������:              �?        ���U�       b�D�	�;����A�*�
w
discriminator_loss*a	   � (?   � (?      �?!   � (?) a)�'>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   @�ܾ   @�ܾ      �?!   @�ܾ)������=2�ѩ�-߾E��a�Wܾ�������:              �?        ���       b�D�	�:����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ����=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   �
�ؾ   �
�ؾ      �?!   �
�ؾ) r�t��=2�iD*L�پ�_�T�l׾�������:              �?        K��       b�D�	 s����A�*�
w
discriminator_loss*a	   �"D�>   �"D�>      �?!   �"D�>) ^�;�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)@\V����=2���%ᾙѩ�-߾�������:              �?        ����       b�D�	ܧ����A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)�0_B�@>2�T7��?�vV�R9?�������:              �?        
s
generator_loss*a	   �$�   �$�      �?!   �$�)@����\�=2���%ᾙѩ�-߾�������:              �?        ����       b�D�	7�����A�*�
w
discriminator_loss*a	   �	�>   �	�>      �?!   �	�>) �긤� >2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   �^Ծ   �^Ծ      �?!   �^Ծ)@А{F��=2��>M|Kվ��~]�[Ӿ�������:              �?        �c��       b�D�	"�����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) d'�7��=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   `�Ծ   `�Ծ      �?!   `�Ծ)@.��FQ�=2��>M|Kվ��~]�[Ӿ�������:              �?        -�f�       b�D�	Ɵ ���A�*�
w
discriminator_loss*a	    �Y ?    �Y ?      �?!    �Y ?)  ��>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ט���=2a�Ϭ(���(����������:              �?        -B=?�       b�D�	�:���A�*�
w
discriminator_loss*a	   �.��>   �.��>      �?!   �.��>) ��|]�>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	   �xھ   �xھ      �?!   �xھ)������=2E��a�Wܾ�iD*L�پ�������:              �?        6�Z�       b�D�	�{	���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �����=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   ��O�   ��O�      �?!   ��O�) I\��=28K�ߝ�a�Ϭ(��������:              �?        Q����       b�D�	�k���A�*�
w
discriminator_loss*a	    �7�>    �7�>      �?!    �7�>) @n�T#�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   ��>�   ��>�      �?!   ��>�)@�H�N��=2})�l a��ߊ4F���������:              �?        � n�       b�D�	pV���A�*�
w
discriminator_loss*a	   �g��>   �g��>      �?!   �g��>)@`��hD�=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) "P���=2���%ᾙѩ�-߾�������:              �?        �$c�       b�D�	�H���A�*�
w
discriminator_loss*a	   @2M?   @2M?      �?!   @2M?)��"*�)>2f�ʜ�7
?>h�'�?�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)�Hx�Ŏ�=2a�Ϭ(���(����������:              �?        �ְ��       b�D�	^5���A�*�
w
discriminator_loss*a	    L�>    L�>      �?!    L�>) &�i���=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	    �ξ    �ξ      �?!    �ξ)  �AG8�=2K+�E��Ͼ['�?�;�������:              �?        ��c��       b�D�	>{=����A�*�
w
discriminator_loss*a	   ��8?   ��8?      �?!   ��8?)@ض��>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  IJ%��=28K�ߝ�a�Ϭ(��������:              �?        �Qa�       b�D�	c�A����A�*�
w
discriminator_loss*a	   ��H�>   ��H�>      �?!   ��H�>) ":���=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   �a�   �a�      �?!   �a�)@�!���=2�uE���⾮��%��������:              �?        ^����       b�D�	�F����A�*�
w
discriminator_loss*a	    4�>    4�>      �?!    4�>) �T�&�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   @�־   @�־      �?!   @�־)�4�/q�=2�_�T�l׾��>M|Kվ�������:              �?        cu O�       b�D�	�J����A�*�
w
discriminator_loss*a	   �E��>   �E��>      �?!   �E��>)���* >2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   @dM�   @dM�      �?!   @dM�) !\J�=2��(��澢f�����������:              �?        �
Ĥ�       b�D�	�)N����A�*�
w
discriminator_loss*a	    s>�>    s>�>      �?!    s>�>) �z��4�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   �sھ   �sھ      �?!   �sھ)�t1����=2E��a�Wܾ�iD*L�پ�������:              �?        �|�       b�D�	�DR����A�*�
w
discriminator_loss*a	   @�}�>   @�}�>      �?!   @�}�>)��5���=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) �>��=2a�Ϭ(���(����������:              �?        X����       b�D�	<bV����A�*�
w
discriminator_loss*a	   ��
�>   ��
�>      �?!   ��
�>) g��W�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	    	�Ӿ    	�Ӿ      �?!    	�Ӿ)@4A����=2��>M|Kվ��~]�[Ӿ�������:              �?        �vx��       b�D�	kQZ����A�*�
w
discriminator_loss*a	    + �>    + �>      �?!    + �>) �s f��=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   @5��   @5��      �?!   @5��)��$=��=28K�ߝ�a�Ϭ(��������:              �?        ��U�       b�D�	On�����A�*�
w
discriminator_loss*a	   �  ?   �  ?      �?!   �  ?)@��# >2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   �8Ӿ   �8Ӿ      �?!   �8Ӿ) ĉ��=2��~]�[Ӿjqs&\�Ѿ�������:              �?        �ǰ2�       b�D�	F������A�*�
w
discriminator_loss*a	   ��7�>   ��7�>      �?!   ��7�>)@�*x��=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	    �ؾ    �ؾ      �?!    �ؾ) LW�z\�=2�iD*L�پ�_�T�l׾�������:              �?        ��e��       b�D�	~<�����A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) f�b
�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   ��^�   ��^�      �?!   ��^�)@6Ur�r�=2�f�����uE�����������:              �?        ����       b�D�	(������A�*�
w
discriminator_loss*a	    ̶�>    ̶�>      �?!    ̶�>)  )k�u�=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	    ؾ    ؾ      �?!    ؾ)  O>0!�=2�iD*L�پ�_�T�l׾�������:              �?        :���       b�D�	=~�����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ҐR,�=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  ��\9�=2��(��澢f�����������:              �?        /��       b�D�	�������A�*�
w
discriminator_loss*a	   ��M�>   ��M�>      �?!   ��M�>)�r��K�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   @&߾   @&߾      �?!   @&߾)��y���=2�ѩ�-߾E��a�Wܾ�������:              �?        r���       b�D�	�������A�*�
w
discriminator_loss*a	   @�y�>   @�y�>      �?!   @�y�>) �=Y�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	    /=ؾ    /=ؾ      �?!    /=ؾ) u{;\�=2�iD*L�پ�_�T�l׾�������:              �?        W���       b�D�	{�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�x0�t��=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@�2��=2���%ᾙѩ�-߾�������:              �?        �P;��       b�D�	������A�*�
w
discriminator_loss*a	   @ �>   @ �>      �?!   @ �>)��0U\�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   ��Ӿ   ��Ӿ      �?!   ��Ӿ) �}F�ܷ=2��>M|Kվ��~]�[Ӿ�������:              �?        ���       b�D�	������A�*�
w
discriminator_loss*a	   `T��>   `T��>      �?!   `T��>)@�\���=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) vCU�S�=2�h���`�8K�ߝ��������:              �?        @�D�       b�D�	I%�����A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ��+�b�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) �\[�b�=2���%ᾙѩ�-߾�������:              �?        U ��       b�D�	.c�����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�ѭ'�=2�f����>��(���>�������:              �?        
s
generator_loss*a	    �T�    �T�      �?!    �T�)@t�� �=2�uE���⾮��%��������:              �?        �!�+�       b�D�	*_�����A�*�
w
discriminator_loss*a	    f	�>    f	�>      �?!    f	�>) @J�Y�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ɫd��=2�uE���⾮��%��������:              �?        �{z�       b�D�	�R�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@N�g�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   �#t۾   �#t۾      �?!   �#t۾)��Wס��=2E��a�Wܾ�iD*L�پ�������:              �?        ��~�       b�D�	K�����A�*�
w
discriminator_loss*a	   ೟�>   ೟�>      �?!   ೟�>)@0�B��=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   @2�   @2�      �?!   @2�) э`7.�=2})�l a��ߊ4F���������:              �?        ψ=_�       b�D�	�p�����A�*�
w
discriminator_loss*a	   �v�>   �v�>      �?!   �v�>)��/,���=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   �
�Ѿ   �
�Ѿ      �?!   �
�Ѿ) �&�ɣ�=2��~]�[Ӿjqs&\�Ѿ�������:              �?        ���       b�D�	�sn����A�*�
w
discriminator_loss*a	    @p�>    @p�>      �?!    @p�>)   ����=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	    �	�    �	�      �?!    �	�) B�'�=2���%ᾙѩ�-߾�������:              �?        7��o�       b�D�	�<t����A�*�
w
discriminator_loss*a	   `�j�>   `�j�>      �?!   `�j�>)@
m)
�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   �+h�   �+h�      �?!   �+h�) D��a�=2��(��澢f�����������:              �?        ��k��       b�D�	%�y����A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@*���*�=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   @{־   @{־      �?!   @{־) i&#>��=2�_�T�l׾��>M|Kվ�������:              �?        U��       b�D�	����A�*�
w
discriminator_loss*a	   �i��>   �i��>      �?!   �i��>) ��<�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   `7�   `7�      �?!   `7�)@���n�=2���%ᾙѩ�-߾�������:              �?        .�]��       b�D�	L������A�*�
w
discriminator_loss*a	   �(��>   �(��>      �?!   �(��>)��/Xw>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   `�ܾ   `�ܾ      �?!   `�ܾ) ՀF�l�=2�ѩ�-߾E��a�Wܾ�������:              �?        t���       b�D�	������A�*�
w
discriminator_loss*a	    J�>    J�>      �?!    J�>) @�0���=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   �+�پ   �+�پ      �?!   �+�پ) (,_�=2�iD*L�پ�_�T�l׾�������:              �?        ��]�       b�D�	������A�*�
w
discriminator_loss*a	   �� ?   �� ?      �?!   �� ?)@tW�$->2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   ��پ   ��پ      �?!   ��پ) XV����=2�iD*L�پ�_�T�l׾�������:              �?        /�+��       b�D�	�8�����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �<͛�=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	    �Ծ    �Ծ      �?!    �Ծ) @�싉�=2��>M|Kվ��~]�[Ӿ�������:              �?        ���+�       b�D�	�� ����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �XgkR�=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   �!�ݾ   �!�ݾ      �?!   �!�ݾ) �����=2�ѩ�-߾E��a�Wܾ�������:              �?        ��)E�       b�D�	r����A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) 98C ��=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   @�о   @�о      �?!   @�о) �DBʱ=2jqs&\�ѾK+�E��Ͼ�������:              �?        ����       b�D�	�O����A�*�
w
discriminator_loss*a	   `gD�>   `gD�>      �?!   `gD�>)@�K�@��=2�f����>��(���>�������:              �?        
s
generator_loss*a	   `վ   `վ      �?!   `վ)@�5r��=2��>M|Kվ��~]�[Ӿ�������:              �?        �����       �N�	#����A*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@��� [�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   ��ؾ   ��ؾ      �?!   ��ؾ) e3���=2�iD*L�پ�_�T�l׾�������:              �?        ~�[�       �{�	�8�����A*�
w
discriminator_loss*a	    #q�>    #q�>      �?!    #q�>) HV�h��=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   �"Iؾ   �"Iؾ      �?!   �"Iؾ)���F[n�=2�iD*L�پ�_�T�l׾�������:              �?        �P�       �{�	O8�����A
*�
w
discriminator_loss*a	   �N��>   �N��>      �?!   �N��>)����_t�=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   ��Ҿ   ��Ҿ      �?!   ��Ҿ) d5���=2��~]�[Ӿjqs&\�Ѿ�������:              �?        ANr�       �{�	������A*�
w
discriminator_loss*a	    �=�>    �=�>      �?!    �=�>) H��u��=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	    �W�    �W�      �?!    �W�)@,*{Zb�=2�f�����uE�����������:              �?        �bx��       �{�	:������A*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)������=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   ��Ҿ   ��Ҿ      �?!   ��Ҿ)@И���=2��~]�[Ӿjqs&\�Ѿ�������:              �?        )�R�       �{�	�'�����A*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) Y.��=2�f����>��(���>�������:              �?        
s
generator_loss*a	    ]۾    ]۾      �?!    ]۾) �u,f�=2E��a�Wܾ�iD*L�پ�������:              �?        J�X(�       �{�	�#�����A*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) �C�U�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   @�ؾ   @�ؾ      �?!   @�ؾ)��7�\��=2�iD*L�پ�_�T�l׾�������:              �?        ��+i�       �{�	LI�����A#*�
w
discriminator_loss*a	    �9�>    �9�>      �?!    �9�>) B~��=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   `
VӾ   `
VӾ      �?!   `
VӾ)@��S'^�=2��~]�[Ӿjqs&\�Ѿ�������:              �?        t��d�       �{�	7������A(*�
w
discriminator_loss*a	     3�>     3�>      �?!     3�>)  @\G}�=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) ��x��=2�uE���⾮��%��������:              �?        �2l�       �{�	7�i����A-*�
w
discriminator_loss*a	    ,^�>    ,^�>      �?!    ,^�>)  yE�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   �Ӿ   �Ӿ      �?!   �Ӿ)@����=2��~]�[Ӿjqs&\�Ѿ�������:              �?        @;�B�       �{�	��m����A2*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@Nސ�}�=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   �+�ھ   �+�ھ      �?!   �+�ھ)��?�'�=2E��a�Wܾ�iD*L�پ�������:              �?        ���       �{�	��q����A7*�
w
discriminator_loss*a	   @�� ?   @�� ?      �?!   @�� ?) �z#mQ>2ji6�9�?�S�F !?�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) ��2q�=2���%ᾙѩ�-߾�������:              �?        )� ��       �{�	�u����A<*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ���D��=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	    .�ݾ    .�ݾ      �?!    .�ݾ) |t	"~�=2�ѩ�-߾E��a�Wܾ�������:              �?        �����       �{�	�Vy����AA*�
w
discriminator_loss*a	   �:��>   �:��>      �?!   �:��>) ��qk�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   �7��   �7��      �?!   �7��)@ wS�+�=2���%ᾙѩ�-߾�������:              �?        ���W�       �{�	_F}����AF*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@�v��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   @7�ھ   @7�ھ      �?!   @7�ھ)�d7��=2E��a�Wܾ�iD*L�پ�������:              �?        �{-�       �{�	�V�����AK*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@Zj@!��=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   �� �   �� �      �?!   �� �) �����=28K�ߝ�a�Ϭ(��������:              �?        5��       �{�	�A�����AP*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ��>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   ��˾   ��˾      �?!   ��˾) �FO�=2['�?�;;�"�qʾ�������:              �?        �!�0�       �{�	�������AU*�
w
discriminator_loss*a	   �s��>   �s��>      �?!   �s��>) �!d���=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   �$fܾ   �$fܾ      �?!   �$fܾ) ���4�=2�ѩ�-߾E��a�Wܾ�������:              �?        vUU�       �{�	 �����AZ*�
w
discriminator_loss*a	   �^?   �^?      �?!   �^?)@���[4>2��d�r?�5�i}1?�������:              �?        
s
generator_loss*a	    5�ݾ    5�ݾ      �?!    5�ݾ) 2�k� �=2�ѩ�-߾E��a�Wܾ�������:              �?        r����       �{�	O����A_*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)����f�>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   ��Ҿ   ��Ҿ      �?!   ��Ҿ)@����=2��~]�[Ӿjqs&\�Ѿ�������:              �?        D'���       �{�	�����Ad*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@6��K�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   �f׾   �f׾      �?!   �f׾) B�}�=2�_�T�l׾��>M|Kվ�������:              �?        W��,�       �{�	 �����Ai*�
w
discriminator_loss*a	   ��[�>   ��[�>      �?!   ��[�>) 2LH��=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	    �Ӿ    �Ӿ      �?!    �Ӿ) �V�CӸ=2��>M|Kվ��~]�[Ӿ�������:              �?        8&���       �{�	�:����An*�
w
discriminator_loss*a	   @�,�>   @�,�>      �?!   @�,�>) i«٤�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   @(�ؾ   @(�ؾ      �?!   @(�ؾ)��@HE�=2�iD*L�پ�_�T�l׾�������:              �?        ���b�       �{�	$�����As*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) kFuuL�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   ��(�   ��(�      �?!   ��(�) Z��(b�=28K�ߝ�a�Ϭ(��������:              �?        ��Oj�       �{�	�����Ax*�
w
discriminator_loss*a	   ��u�>   ��u�>      �?!   ��u�>) `\�i��=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   `vѾ   `vѾ      �?!   `vѾ)@.SqC�=2jqs&\�ѾK+�E��Ͼ�������:              �?        #�b��       �{�	��� ���A}*�
w
discriminator_loss*a	    %��>    %��>      �?!    %��>) O�/�=2�f����>��(���>�������:              �?        
s
generator_loss*a	    %�Ӿ    %�Ӿ      �?!    %�Ӿ)@$�Oڷ=2��>M|Kվ��~]�[Ӿ�������:              �?        �2�J�       b�D�	�@� ���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) @^��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   ��Ҿ   ��Ҿ      �?!   ��Ҿ)@�����=2��~]�[Ӿjqs&\�Ѿ�������:              �?        /�:��       b�D�	U�� ���A�*�
w
discriminator_loss*a	   ��W�>   ��W�>      �?!   ��W�>) �_����=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   ��پ   ��پ      �?!   ��پ) un���=2�iD*L�پ�_�T�l׾�������:              �?        c	���       b�D�	��� ���A�*�
w
discriminator_loss*a	    L8�>    L8�>      �?!    L8�>) ��
���=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   @hؾ   @hؾ      �?!   @hؾ)�47�{��=2�iD*L�پ�_�T�l׾�������:              �?        �E�       b�D�	�� ���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@b�I��=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   ��ľ   ��ľ      �?!   ��ľ) ��%�]�=2����ž�XQ�þ�������:              �?        �����       b�D�	Bh� ���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  $ޠ��=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   �"�۾   �"�۾      �?!   �"�۾) 2U�Q��=2E��a�Wܾ�iD*L�پ�������:              �?        ͒�U�       b�D�	��� ���A�*�
w
discriminator_loss*a	   @5�>   @5�>      �?!   @5�>)��	�:o�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	    �)�    �)�      �?!    �)�)@\rh��=2�f�����uE�����������:              �?        ��v�       b�D�	�)� ���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ��%���=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   ��&�   ��&�      �?!   ��&�) �Q~�M�=2���%ᾙѩ�-߾�������:              �?        ��?�       b�D�	fi����A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) @P����=2�f����>��(���>�������:              �?        
s
generator_loss*a	   @8�׾   @8�׾      �?!   @8�׾)��6�8a�=2�iD*L�پ�_�T�l׾�������:              �?        ��u�       b�D�	Q����A�*�
w
discriminator_loss*a	   �cd�>   �cd�>      �?!   �cd�>)@R�{��=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   �bA�   �bA�      �?!   �bA�)@�����=2�uE���⾮��%��������:              �?        ����       b�D�	}
����A�*�
w
discriminator_loss*a	    ?P�>    ?P�>      �?!    ?P�>) ��?��=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   �h۾   �h۾      �?!   �h۾) ��y�=2E��a�Wܾ�iD*L�پ�������:              �?        �Py�       b�D�	Ĺ����A�*�
w
discriminator_loss*a	   �A�>   �A�>      �?!   �A�>) �	>6�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   �+Ӿ   �+Ӿ      �?!   �+Ӿ) ��L���=2��~]�[Ӿjqs&\�Ѿ�������:              �?        ��c+�       b�D�	�٨���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@i[��=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	    Ӿ    Ӿ      �?!    Ӿ)@l�1���=2��~]�[Ӿjqs&\�Ѿ�������:              �?        �g�P�       b�D�	Aڭ���A�*�
w
discriminator_loss*a	   � ��>   � ��>      �?!   � ��>)@�����=2�f����>��(���>�������:              �?        
s
generator_loss*a	   �X۾   �X۾      �?!   �X۾) BQ)�]�=2E��a�Wܾ�iD*L�پ�������:              �?        u����       b�D�	������A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ]��R<�=2�h���`�>�ߊ4F��>�������:              �?        
s
generator_loss*a	   ��d�   ��d�      �?!   ��d�) b1E�=2a�Ϭ(���(����������:              �?        �Q��       b�D�	�.����A�*�
w
discriminator_loss*a	   �;��>   �;��>      �?!   �;��>) !��R�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	    �Ѿ    �Ѿ      �?!    �Ѿ) @�b�c�=2��~]�[Ӿjqs&\�Ѿ�������:              �?        ��C��       b�D�	c�4���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �-9���=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �
�̾   �
�̾      �?!   �
�̾) rs*��=2['�?�;;�"�qʾ�������:              �?        ��3��       b�D�	�29���A�*�
w
discriminator_loss*a	   ��}?   ��}?      �?!   ��}?)@G�>:>2�5�i}1?�T7��?�������:              �?        
s
generator_loss*a	   @'�׾   @'�׾      �?!   @'�׾)�$���C�=2�iD*L�پ�_�T�l׾�������:              �?        &0�       b�D�	Ms=���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@�>(48�=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   �&hҾ   �&hҾ      �?!   �&hҾ) �(�,�=2��~]�[Ӿjqs&\�Ѿ�������:              �?        ���T�       b�D�	݄A���A�*�
w
discriminator_loss*a	   �4�>   �4�>      �?!   �4�>) )�jD�=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   ��̾   ��̾      �?!   ��̾) �7L�=2['�?�;;�"�qʾ�������:              �?        �dj��       b�D�	P�E���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) Y�шb�=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   �&�ؾ   �&�ؾ      �?!   �&�ؾ) :}�L��=2�iD*L�پ�_�T�l׾�������:              �?        ^&�       b�D�	�JJ���A�*�
w
discriminator_loss*a	   �/��>   �/��>      �?!   �/��>) �v.���=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   �,�پ   �,�پ      �?!   �,�پ) ���[�=2�iD*L�پ�_�T�l׾�������:              �?        ��
n�       b�D�	ݑN���A�*�
w
discriminator_loss*a	   @�*�>   @�*�>      �?!   @�*�>) ���U�=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   `A־   `A־      �?!   `A־)@"1���=2�_�T�l׾��>M|Kվ�������:              �?        4�t��       b�D�	l�R���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) @�����=2�ߊ4F��>})�l a�>�������:              �?        
s
generator_loss*a	   `Rؾ   `Rؾ      �?!   `Rؾ) )�A�{�=2�iD*L�پ�_�T�l׾�������:              �?        jyz��       b�D�	����A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) 3	~0�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   �"�Ծ   �"�Ծ      �?!   �"�Ծ)@�
�jf�=2��>M|Kվ��~]�[Ӿ�������:              �?        e�t��       b�D�	[]���A�*�
w
discriminator_loss*a	   �f�>   �f�>      �?!   �f�>)�,�g��=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   @:Ծ   @:Ծ      �?!   @:Ծ) ��r��=2��>M|Kվ��~]�[Ӿ�������:              �?        �j��       b�D�	����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��0�e�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   �jѾ   �jѾ      �?!   �jѾ)@�Q���=2jqs&\�ѾK+�E��Ͼ�������:              �?        �?���       b�D�	�3���A�*�
w
discriminator_loss*a	   �#}�>   �#}�>      �?!   �#}�>) �yΞ��=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   @z̾   @z̾      �?!   @z̾)��vW�=2['�?�;;�"�qʾ�������:              �?        �i��       b�D�	2����A�*�
w
discriminator_loss*a	   @ų�>   @ų�>      �?!   @ų�>)��{sH�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   �jپ   �jپ      �?!   �jپ) ��./�=2�iD*L�پ�_�T�l׾�������:              �?        \����       b�D�	2m���A�*�
w
discriminator_loss*a	   ��5�>   ��5�>      �?!   ��5�>) D�z���=2�f����>��(���>�������:              �?        
s
generator_loss*a	    �Wݾ    �Wݾ      �?!    �Wݾ) H���=2�ѩ�-߾E��a�Wܾ�������:              �?        ~�&�       b�D�	�) ���A�*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) �<�t��=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	    �;    �;      �?!    �;) F�h}�=2K+�E��Ͼ['�?�;�������:              �?        Tw=�       b�D�	�+%���A�*�
w
discriminator_loss*a	   `�>   `�>      �?!   `�>) G �L�=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   �	�о   �	�о      �?!   �	�о) �=-$�=2jqs&\�ѾK+�E��Ͼ�������:              �?        �f��       b�D�	Y{����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  �Ŕ�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �
о   �
о      �?!   �
о)@�)$�=2jqs&\�ѾK+�E��Ͼ�������:              �?        ��c9�       b�D�	�V����A�*�
w
discriminator_loss*a	   ��'�>   ��'�>      �?!   ��'�>) j{�^>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	    �Ծ    �Ծ      �?!    �Ծ)  ��-��=2��>M|Kվ��~]�[Ӿ�������:              �?        !��W�       b�D�	eA����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ���{��=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   �	.ɾ   �	.ɾ      �?!   �	.ɾ) g�1У=2;�"�qʾ
�/eq
Ⱦ�������:              �?        8o0:�       b�D�	�I����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@��s)�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   �3ݾ   �3ݾ      �?!   �3ݾ)��?�{n�=2�ѩ�-߾E��a�Wܾ�������:              �?        qvR��       b�D�	�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@N��U�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	    
jȾ    
jȾ      �?!    
jȾ)  Cbn��=2;�"�qʾ
�/eq
Ⱦ�������:              �?        �s�o�       b�D�	����A�*�
w
discriminator_loss*a	    S� ?    S� ?      �?!    S� ?) �nj'�>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	   ��ɾ   ��ɾ      �?!   ��ɾ)���i,��=2;�"�qʾ
�/eq
Ⱦ�������:              �?        c��_�       b�D�	�����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)���=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	   �$ʾ   �$ʾ      �?!   �$ʾ) ����Z�=2;�"�qʾ
�/eq
Ⱦ�������:              �?        e�.V�       b�D�	������A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@���.�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   @�־   @�־      �?!   @�־)��j*g�=2�_�T�l׾��>M|Kվ�������:              �?        -!���       b�D�	�^~	���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�(��:8�=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	    VϾ    VϾ      �?!    VϾ) �U���=2K+�E��Ͼ['�?�;�������:              �?        ����       b�D�	�W�	���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��y&g�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   �	Ծ   �	Ծ      �?!   �	Ծ) q!���=2��>M|Kվ��~]�[Ӿ�������:              �?        -m���       b�D�	@T�	���A�*�
w
discriminator_loss*a	   @l�>   @l�>      �?!   @l�>) ai�y��=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	    ,Ͼ    ,Ͼ      �?!    ,Ͼ) n�¡]�=2K+�E��Ͼ['�?�;�������:              �?        b�e��       b�D�	lJ�	���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �7,\�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	    �ľ    �ľ      �?!    �ľ)@Xr� ��=2����ž�XQ�þ�������:              �?        ˛��       b�D�	S�	���A�*�
w
discriminator_loss*a	   �%�>   �%�>      �?!   �%�>)����K�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   ��о   ��о      �?!   ��о) )Z�~c�=2jqs&\�ѾK+�E��Ͼ�������:              �?        �H���       b�D�	��	���A�*�
w
discriminator_loss*a	    A\�>    A\�>      �?!    A\�>)@�����=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   `�о   `�о      �?!   `�о)@vV|H�=2jqs&\�ѾK+�E��Ͼ�������:              �?        ��>p�       b�D�	���	���A�*�
w
discriminator_loss*a	    !��>    !��>      �?!    !��>) r�M��=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   @ �پ   @ �پ      �?!   @ �پ)���-���=2E��a�Wܾ�iD*L�پ�������:              �?        ����       b�D�	���	���A�*�
w
discriminator_loss*a	   �Xo�>   �Xo�>      �?!   �Xo�>) ֈ3�)>2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@�^���=2�uE���⾮��%��������:              �?        �����       b�D�	Vy!���A�*�
w
discriminator_loss*a	    �?    �?      �?!    �?)@�O �0>2x?�x�?��d�r?�������:              �?        
s
generator_loss*a	   @ �׾   @ �׾      �?!   @ �׾)�����e�=2�iD*L�پ�_�T�l׾�������:              �?        ��J�       b�D�	�&���A�*�
w
discriminator_loss*a	   �K�>   �K�>      �?!   �K�>) $*Ne��=2���%�>�uE����>�������:              �?        
s
generator_loss*a	    �о    �о      �?!    �о) N����=2jqs&\�ѾK+�E��Ͼ�������:              �?        �(]�       b�D�	d�+���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �4}`��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   `�ھ   `�ھ      �?!   `�ھ) oy�4�=2E��a�Wܾ�iD*L�پ�������:              �?        �|t�       b�D�	��0���A�*�
w
discriminator_loss*a	   ��_�>   ��_�>      �?!   ��_�>)@։!n�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   `C�ݾ   `C�ݾ      �?!   `C�ݾ) �۪���=2�ѩ�-߾E��a�Wܾ�������:              �?        �v��       b�D�	��5���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@������=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   @eҾ   @eҾ      �?!   @eҾ) �LR�%�=2��~]�[Ӿjqs&\�Ѿ�������:              �?        c����       b�D�	�o:���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) Hks��=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �Ҿ   �Ҿ      �?!   �Ҿ) OT�=2��~]�[Ӿjqs&\�Ѿ�������:              �?        �5+�       b�D�	�d?���A�*�
w
discriminator_loss*a	   `H��>   `H��>      �?!   `H��>)@b�XA�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   ��ʾ   ��ʾ      �?!   ��ʾ) 9𰽩�=2['�?�;;�"�qʾ�������:              �?        Pس�       b�D�	�ND���A�*�
w
discriminator_loss*a	   @RU ?   @RU ?      �?!   @RU ?) ю{k�>2�FF�G ?��[�?�������:              �?        
s
generator_loss*a	    �Ҿ    �Ҿ      �?!    �Ҿ)@@(8�=2��~]�[Ӿjqs&\�Ѿ�������:              �?        HH���       b�D�	q�����A�*�
w
discriminator_loss*a	   ��`�>   ��`�>      �?!   ��`�>) �y�r��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   ��־   ��־      �?!   ��־) �7�G�=2�_�T�l׾��>M|Kվ�������:              �?        �ƶ4�       b�D�	�����A�*�
w
discriminator_loss*a	   �i��>   �i��>      �?!   �i��>) �<��=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   �d��   �d��      �?!   �d��) i:���=2�f�����uE�����������:              �?        �D��       b�D�	}�����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) R����=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   @�о   @�о      �?!   @�о) ���3B�=2jqs&\�ѾK+�E��Ͼ�������:              �?        h�.�       b�D�	�9����A�*�
w
discriminator_loss*a	   �g�>   �g�>      �?!   �g�>)@�wgv*�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   `�о   `�о      �?!   `�о)@.�~r�=2jqs&\�ѾK+�E��Ͼ�������:              �?        >H}��       b�D�	�h ���A�*�
w
discriminator_loss*a	   @m�>   @m�>      �?!   @m�>) �4��=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   ��Ͼ   ��Ͼ      �?!   ��Ͼ) ��!m��=2K+�E��Ͼ['�?�;�������:              �?        ̃��       b�D�	�M���A�*�
w
discriminator_loss*a	    &!�>    &!�>      �?!    &!�>)@؞�m��=2�f����>��(���>�������:              �?        
s
generator_loss*a	   ` �پ   ` �پ      �?!   ` �پ) ��%��=2�iD*L�پ�_�T�l׾�������:              �?        }u
2�       b�D�	�?���A�*�
w
discriminator_loss*a	   ��r�>   ��r�>      �?!   ��r�>)@�\y��=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   @�վ   @�վ      �?!   @�վ) ����=2�_�T�l׾��>M|Kվ�������:              �?        ���$�       b�D�	_����A�*�
w
discriminator_loss*a	   `l��>   `l��>      �?!   `l��>)@~ |�=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   ��;   ��;      �?!   ��;) ��� �=2K+�E��Ͼ['�?�;�������:              �?        �B}T�       b�D�	������A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) Dk�Y0�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	    z;    z;      �?!    z;) ��.'�=2K+�E��Ͼ['�?�;�������:              �?        9�p>�       b�D�	������A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �\��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   `?о   `?о      �?!   `?о)@�}1�=2jqs&\�ѾK+�E��Ͼ�������:              �?        ���m�       b�D�	�����A�*�
w
discriminator_loss*a	    w�>    w�>      �?!    w�>)@���^O�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   �"þ   �"þ      �?!   �"þ) ����=2�XQ�þ��~��¾�������:              �?        n.��       b�D�	�`����A�*�
w
discriminator_loss*a	   �?��>   �?��>      �?!   �?��>) n*���=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	    �Ǿ    �Ǿ      �?!    �Ǿ) ǚ�_�=2
�/eq
Ⱦ����ž�������:              �?        �]5g�       b�D�	&�����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@L�2�ȿ=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   `	ɾ   `	ɾ      �?!   `	ɾ) ���}��=2;�"�qʾ
�/eq
Ⱦ�������:              �?        x�$��       b�D�	������A�*�
w
discriminator_loss*a	   �)��>   �)��>      �?!   �)��>) ��*u�=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	    �ʾ    �ʾ      �?!    �ʾ) n�1���=2['�?�;;�"�qʾ�������:              �?        ;�b�       b�D�	h�����A�*�
w
discriminator_loss*a	   @"��>   @"��>      �?!   @"��>) QR���=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   `Hо   `Hо      �?!   `Hо)@bdU��=2jqs&\�ѾK+�E��Ͼ�������:              �?        7M���       b�D�	�v����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@pa��=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   `	rǾ   `	rǾ      �?!   `	rǾ) ��܃-�=2
�/eq
Ⱦ����ž�������:              �?        g!��       b�D�	�^����A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�|�m\�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �4Ѿ   �4Ѿ      �?!   �4Ѿ) �DG;�=2jqs&\�ѾK+�E��Ͼ�������:              �?        UQ�d�       b�D�	�U����A�*�
w
discriminator_loss*a	     d�>     d�>      �?!     d�>)  @��U�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	    �о    �о      �?!    �о) �m�Nt�=2jqs&\�ѾK+�E��Ͼ�������:              �?        ��U�       b�D�	�a����A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)���&]�=28K�ߝ�>�h���`�>�������:              �?        
s
generator_loss*a	    
x��    
x��      �?!    
x��)  �U�=2�[�=�k���*��ڽ��������:              �?        +a.��       b�D�	h^����A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) ����I�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   ��ž   ��ž      �?!   ��ž) )J'*�=2
�/eq
Ⱦ����ž�������:              �?        ޼_~�       b�D�	�#����A�*�
w
discriminator_loss*a	   �'�>   �'�>      �?!   �'�>)@"g7�=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	    8�־    8�־      �?!    8�־) pƼ��=2�_�T�l׾��>M|Kվ�������:              �?        �1%��       b�D�	�����A�*�
w
discriminator_loss*a	    X�>    X�>      �?!    X�>) *9z�>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   ��ξ   ��ξ      �?!   ��ξ)��:h�V�=2K+�E��Ͼ['�?�;�������:              �?        d����       b�D�	U©���A�*�
w
discriminator_loss*a	    	[�>    	[�>      �?!    	[�>) e���=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   @ľ   @ľ      �?!   @ľ) q���=2����ž�XQ�þ�������:              �?        �|���       b�D�	_�����A�*�
w
discriminator_loss*a	    	N�>    	N�>      �?!    	N�>) ����=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   @�   @�      �?!   @�)��˜)d�=2�*��ڽ�G&�$���������:              �?        z���       b�D�	+�K���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) ��ƶ=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   @
Ǿ   @
Ǿ      �?!   @
Ǿ)�H�����=2
�/eq
Ⱦ����ž�������:              �?        �L���       b�D�	m�O���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ����=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   ��Ӿ   ��Ӿ      �?!   ��Ӿ)@�v�~�=2��>M|Kվ��~]�[Ӿ�������:              �?        �~7�       b�D�	��S���A�*�
w
discriminator_loss*a	   `9��>   `9��>      �?!   `9��>) ߾�4H�=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   `	�ľ   `	�ľ      �?!   `	�ľ)@~eZ���=2����ž�XQ�þ�������:              �?        {���       b�D�	��W���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) ��,�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   �8Ҿ   �8Ҿ      �?!   �8Ҿ) �Q%꾴=2��~]�[Ӿjqs&\�Ѿ�������:              �?        c�%�       b�D�	�q[���A�*�
w
discriminator_loss*a	   �Q��>   �Q��>      �?!   �Q��>)�����=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   `$�о   `$�о      �?!   `$�о)@��b�a�=2jqs&\�ѾK+�E��Ͼ�������:              �?        P�H��       b�D�	=8_���A�*�
w
discriminator_loss*a	   @		�>   @		�>      �?!   @		�>) Ym�YT�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   @~¾   @~¾      �?!   @~¾) !�j_�=2�XQ�þ��~��¾�������:              �?        �����       b�D�	�c���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ��a��=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) ��!�>�=2��~��¾�[�=�k���������:              �?        !:���       b�D�	�]g���A�*�
w
discriminator_loss*a	    5v�>    5v�>      �?!    5v�>) 2$X��=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �4־   �4־      �?!   �4־)@�ґK�=2�_�T�l׾��>M|Kվ�������:              �?        y+���       b�D�	X�'���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@�Y��=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   �Bľ   �Bľ      �?!   �Bľ) ��=#��=2����ž�XQ�þ�������:              �?        �X	�       b�D�	�,���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �'\q�=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   ��Ⱦ   ��Ⱦ      �?!   ��Ⱦ) ���h�=2;�"�qʾ
�/eq
Ⱦ�������:              �?        ExwW�       b�D�	Q1���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��%��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   ��yܾ   ��yܾ      �?!   ��yܾ)�4<��V�=2�ѩ�-߾E��a�Wܾ�������:              �?        ����       b�D�	w�7���A�*�
w
discriminator_loss*a	    (U�>    (U�>      �?!    (U�>)  �����=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �
�þ   �
�þ      �?!   �
�þ) �XR��=2����ž�XQ�þ�������:              �?        d���       b�D�	�_<���A�*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>)@r� }��=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   @�;   @�;      �?!   @�;)���b��=2K+�E��Ͼ['�?�;�������:              �?        풅��       b�D�	$/A���A�*�
w
discriminator_loss*a	   �h�>   �h�>      �?!   �h�>) �ə��=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   �xǾ   �xǾ      �?!   �xǾ)�P$<S6�=2
�/eq
Ⱦ����ž�������:              �?        K�n�       b�D�	�F���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�0`:���=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �ƾ   �ƾ      �?!   �ƾ) �s��q�=2
�/eq
Ⱦ����ž�������:              �?        m噦�       b�D�	-�J���A�*�
w
discriminator_loss*a	   �-��>   �-��>      �?!   �-��>) ċ����=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	    Jƾ    Jƾ      �?!    Jƾ)  ���=2
�/eq
Ⱦ����ž�������:              �?        ;R���       b�D�	���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) D���K�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   ��¾   ��¾      �?!   ��¾) �dN��=2�XQ�þ��~��¾�������:              �?        �[���       b�D�	�����A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�<�eN�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   @6Ӿ   @6Ӿ      �?!   @6Ӿ) �<x�=2��~]�[Ӿjqs&\�Ѿ�������:              �?        ���'�       b�D�	�K
���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ����>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��) �{f�y�=2})�l a��ߊ4F���������:              �?        �-��       b�D�	�C���A�*�
w
discriminator_loss*a	   �XX�>   �XX�>      �?!   �XX�>)@���y�=2�f����>��(���>�������:              �?        
s
generator_loss*a	   `�vܾ   `�vܾ      �?!   `�vܾ) 1	�%Q�=2�ѩ�-߾E��a�Wܾ�������:              �?        ?Ka�       b�D�	����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) d;��Ƹ=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   @�Ǿ   @�Ǿ      �?!   @�Ǿ)�|��a��=2
�/eq
Ⱦ����ž�������:              �?        Ŵ���       b�D�	� ���A�*�
w
discriminator_loss*a	   �*�>   �*�>      �?!   �*�>)������=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   �'�־   �'�־      �?!   �'�־) �h9g�=2�_�T�l׾��>M|Kվ�������:              �?        �%��       b�D�	)���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ��Wh�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   �
�þ   �
�þ      �?!   �
�þ) 9��
f�=2�XQ�þ��~��¾�������:              �?        h�|��       b�D�	3�"���A�*�
w
discriminator_loss*a	   ��^�>   ��^�>      �?!   ��^�>) Č���=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   �"\ʾ   �"\ʾ      �?!   �"\ʾ) �����=2;�"�qʾ
�/eq
Ⱦ�������:              �?        ݣ���       b�D�	Pd����A�*�
w
discriminator_loss*a	   �.��>   �.��>      �?!   �.��>) �#+��=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   ��¾   ��¾      �?!   ��¾) ���U�=2�XQ�þ��~��¾�������:              �?        �f�o�       b�D�	0-����A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) C�޴�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   �佾   �佾      �?!   �佾) B"a��=2�[�=�k���*��ڽ��������:              �?        �;=��       b�D�	:����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �2~��=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   `	�ž   `	�ž      �?!   `	�ž)@~���J�=2����ž�XQ�þ�������:              �?        �_��       b�D�	������A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �;���=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ��y1��=2�[�=�k���*��ڽ��������:              �?        Z>�#�       b�D�	ƥ����A�*�
w
discriminator_loss*a	   �n�>   �n�>      �?!   �n�>) "���=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   @�þ   @�þ      �?!   @�þ) q��	!�=2�XQ�þ��~��¾�������:              �?        ��       b�D�	����A�*�
w
discriminator_loss*a	   �x�>   �x�>      �?!   �x�>) �����=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	    H��    H��      �?!    H��) ��?���=2�[�=�k���*��ڽ��������:              �?        ���       b�D�	\����A�*�
w
discriminator_loss*a	   ��u�>   ��u�>      �?!   ��u�>)�M����=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   ���پ   ���پ      �?!   ���پ)� �	U�=2E��a�Wܾ�iD*L�پ�������:              �?        ���       b�D�	ǀ	���A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)��{�a~�=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   �3��   �3��      �?!   �3��) �����=2�uE���⾮��%��������:              �?        �a��       b�D�	i�����A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)������=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   �Nľ   �Nľ      �?!   �Nľ) �Tu�ę=2����ž�XQ�þ�������:              �?        $]a�       b�D�	"k����A�*�
w
discriminator_loss*a	   @)�>   @)�>      �?!   @)�>)�,��E�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   ��Ⱦ   ��Ⱦ      �?!   ��Ⱦ) ���_u�=2;�"�qʾ
�/eq
Ⱦ�������:              �?        ���       b�D�	�p����A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) $S=T8�=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   �h��   �h��      �?!   �h��)�p`|��=25�"�g���0�6�/n���������:              �?        �����       b�D�	x}����A�*�
w
discriminator_loss*a	    a�>    a�>      �?!    a�>) �D{�=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   ��Ⱦ   ��Ⱦ      �?!   ��Ⱦ)�d���e�=2;�"�qʾ
�/eq
Ⱦ�������:              �?        ���`�       b�D�	jB����A�*�
w
discriminator_loss*a	   �>�>   �>�>      �?!   �>�>)@�y̔=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   @\��   @\��      �?!   @\��) �`E�y=2�u`P+d����n������������:              �?        AZ'��       b�D�	g1����A�*�
w
discriminator_loss*a	   �$w�>   �$w�>      �?!   �$w�>) ��/ɒ�=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   `zɾ   `zɾ      �?!   `zɾ) cC��H�=2;�"�qʾ
�/eq
Ⱦ�������:              �?        �X7�       b�D�	.����A�*�
w
discriminator_loss*a	   ��)�>   ��)�>      �?!   ��)�>) 1"Z�h�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   `d�޾   `d�޾      �?!   `d�޾) �<e�'�=2�ѩ�-߾E��a�Wܾ�������:              �?        �	t8�       b�D�	�����A�*�
w
discriminator_loss*a	   �o�>   �o�>      �?!   �o�>) dŰ�<�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   @#�Ͼ   @#�Ͼ      �?!   @#�Ͼ)�Ԟ��-�=2K+�E��Ͼ['�?�;�������:              �?        )�n�       b�D�	��s���A�*�
w
discriminator_loss*a	   @,��>   @,��>      �?!   @,��>) ajZ��=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	    ,vԾ    ,vԾ      �?!    ,vԾ)  y��*�=2��>M|Kվ��~]�[Ӿ�������:              �?        e(��       b�D�	o�x���A�*�
w
discriminator_loss*a	   �5��>   �5��>      �?!   �5��>)�H�KO>�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   ��;   ��;      �?!   ��;)�0��ax�=2K+�E��Ͼ['�?�;�������:              �?        :�(��       b�D�	�}~���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ���H�=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   @"�Ѿ   @"�Ѿ      �?!   @"�Ѿ) Q�N�ҳ=2��~]�[Ӿjqs&\�Ѿ�������:              �?        g^5�       b�D�	�����A�*�
w
discriminator_loss*a	   `h�>   `h�>      �?!   `h�>) Ռ�	�=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   ��ľ   ��ľ      �?!   ��ľ) ���iQ�=2����ž�XQ�þ�������:              �?        ����       b�D�	�=����A�*�
w
discriminator_loss*a	   ै�>   ै�>      �?!   ै�>) �K�~"�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   @�ž   @�ž      �?!   @�ž) ��ϠJ�=2����ž�XQ�þ�������:              �?        +(��       b�D�	r����A�*�
w
discriminator_loss*a	   �`�>   �`�>      �?!   �`�>) �]Ak�=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   �h��   �h��      �?!   �h��) ����=2��~��¾�[�=�k���������:              �?        B����       b�D�	������A�*�
w
discriminator_loss*a	    e��>    e��>      �?!    e��>) �>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	   �&@ξ   �&@ξ      �?!   �&@ξ)��.Cɘ�=2K+�E��Ͼ['�?�;�������:              �?        �����       b�D�	�D����A�*�
w
discriminator_loss*a	    P�?    P�?      �?!    P�?)  �{U�>21��a˲?6�]��?�������:              �?        
s
generator_loss*a	   �¾   �¾      �?!   �¾) d���_�=2��~��¾�[�=�k���������:              �?        !�J3�       b�D�	0Z���A�*�
w
discriminator_loss*a	   @r�>   @r�>      �?!   @r�>)�|m���=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   ��¾   ��¾      �?!   ��¾) ��o$�=2�XQ�þ��~��¾�������:              �?        �Ь�       b�D�	o�]���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) Z����=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)����!�=2�[�=�k���*��ڽ��������:              �?        �J�o�       b�D�	�a���A�*�
w
discriminator_loss*a	    r�>    r�>      �?!    r�>) @��; �=2�XQ��>�����>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) ��p�V�=2�*��ڽ�G&�$���������:              �?        `,,�       b�D�	�e���A�*�
w
discriminator_loss*a	   �cS�>   �cS�>      �?!   �cS�>) )��}�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)��`l+Ɖ=2�*��ڽ�G&�$���������:              �?        {�	�       b�D�	Ãi���A�*�
w
discriminator_loss*a	   �$n�>   �$n�>      �?!   �$n�>) ��X��=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   @$�ɾ   @$�ɾ      �?!   @$�ɾ)��1ܤ=2;�"�qʾ
�/eq
Ⱦ�������:              �?        k�Z��       b�D�	`�m���A�*�
w
discriminator_loss*a	   �/��>   �/��>      �?!   �/��>) -�N��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) �⽣�=2��~��¾�[�=�k���������:              �?        �<��       b�D�	�q���A�*�
w
discriminator_loss*a	   @N�>   @N�>      �?!   @N�>)�x��	��=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	    �ž    �ž      �?!    �ž)  ��J�=2����ž�XQ�þ�������:              �?        �����       b�D�	�cu���A�*�
w
discriminator_loss*a	    &��>    &��>      �?!    &��>)  �sߎ�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   �$�پ   �$�پ      �?!   �$�پ)�4����=2�iD*L�پ�_�T�l׾�������:              �?        L8_��       b�D�	j�1���A�*�
w
discriminator_loss*a	   �z�>   �z�>      �?!   �z�>)@��dV�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) Dc�g�=2��~��¾�[�=�k���������:              �?        #��       b�D�	�5���A�*�
w
discriminator_loss*a	   �LM�>   �LM�>      �?!   �LM�>)�n��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   @ɾ   @ɾ      �?!   @ɾ)���IH��=2;�"�qʾ
�/eq
Ⱦ�������:              �?        ��Ξ�       b�D�	^�9���A�*�
w
discriminator_loss*a	   �F��>   �F��>      �?!   �F��>)�l�1�2�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   �þ   �þ      �?!   �þ) D(��ז=2�XQ�þ��~��¾�������:              �?        �!�       b�D�	9�=���A�*�
w
discriminator_loss*a	   �U��>   �U��>      �?!   �U��>)������=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   �	ƾ   �	ƾ      �?!   �	ƾ) �eb�E�=2
�/eq
Ⱦ����ž�������:              �?         �"��       b�D�	3�A���A�*�
w
discriminator_loss*a	   @%��>   @%��>      �?!   @%��>)�\�z�Z�=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)�����	�=2�[�=�k���*��ڽ��������:              �?        �Z��       b�D�	��E���A�*�
w
discriminator_loss*a	    ,�>    ,�>      �?!    ,�>) ��L�H�=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	    �ʾ    �ʾ      �?!    �ʾ)  �j��=2['�?�;;�"�qʾ�������:              �?        E��       b�D�	��I���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)���}e��=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	    ,�Ͼ    ,�Ͼ      �?!    ,�Ͼ) �<$a�=2K+�E��Ͼ['�?�;�������:              �?        �G���       b�D�	�cN���A�*�
w
discriminator_loss*a	   �VY�>   �VY�>      �?!   �VY�>) �@��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   �5�̾   �5�̾      �?!   �5�̾) r����=2['�?�;;�"�qʾ�������:              �?        ����       b�D�	��-!���A�*�
w
discriminator_loss*a	    4��>    4��>      �?!    4��>) �&G�2�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �~Ⱦ   �~Ⱦ      �?!   �~Ⱦ) ��a	��=2;�"�qʾ
�/eq
Ⱦ�������:              �?        -��W�       b�D�	�{2!���A�*�
w
discriminator_loss*a	    e��>    e��>      �?!    e��>) ��1���=2�f����>��(���>�������:              �?        
s
generator_loss*a	   ��̾   ��̾      �?!   ��̾) r^�ѩ=2['�?�;;�"�qʾ�������:              �?        jC_~�       b�D�	�j6!���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) m��z�=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   �gԾ   �gԾ      �?!   �gԾ) ��qP�=2��>M|Kվ��~]�[Ӿ�������:              �?        Y��	�       �N�	[�9!���A*�
w
discriminator_loss*a	   @R��>   @R��>      �?!   @R��>) ����n�=2�f����>��(���>�������:              �?        
s
generator_loss*a	    *þ    *þ      �?!    *þ)  jA��=2�XQ�þ��~��¾�������:              �?        0�.g�       �{�	jfT#���A*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �,/�O�=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   @�ž   @�ž      �?!   @�ž) �T�U��=2
�/eq
Ⱦ����ž�������:              �?        !���       �{�	LDX#���A
*�
w
discriminator_loss*a	   �:��>   �:��>      �?!   �:��>) R6B�u�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   `9Ӿ   `9Ӿ      �?!   `9Ӿ)@�]���=2��~]�[Ӿjqs&\�Ѿ�������:              �?        �;�       �{�	�F\#���A*�
w
discriminator_loss*a	   @�X�>   @�X�>      �?!   @�X�>) #� ��=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   `�ž   `�ž      �?!   `�ž)@fcٗ��=2����ž�XQ�þ�������:              �?        ����       �{�	�`#���A*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) =m��=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   `@��   `@��      �?!   `@��) �χ�=25�"�g���0�6�/n���������:              �?        t0�       �{�	�e#���A*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@P�[��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   �;   �;      �?!   �;) �t�Sw�=2K+�E��Ͼ['�?�;�������:              �?        |Z��       �{�	pVi#���A*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�3:��=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   `�ɾ   `�ɾ      �?!   `�ɾ) �"��=2;�"�qʾ
�/eq
Ⱦ�������:              �?        �����       �{�	nm#���A#*�
w
discriminator_loss*a	   �s�>   �s�>      �?!   �s�>) Y(ߑ��=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	    �ʾ    �ʾ      �?!    �ʾ)  ��`��=2['�?�;;�"�qʾ�������:              �?        ����       �{�	OWq#���A(*�
w
discriminator_loss*a	    h��>    h��>      �?!    h��>)  R���=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   �
ɾ   �
ɾ      �?!   �
ɾ)�$��ɗ�=2;�"�qʾ
�/eq
Ⱦ�������:              �?        ��ߘ�       �{�	��N%���A-*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ���l�=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   `侾   `侾      �?!   `侾) E���э=2�[�=�k���*��ڽ��������:              �?        =���       �{�	��R%���A2*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@��sH�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   �"ʾ   �"ʾ      �?!   �"ʾ) ��
�W�=2;�"�qʾ
�/eq
Ⱦ�������:              �?        k#TW�       �{�	��V%���A7*�
w
discriminator_loss*a	    0)�>    0)�>      �?!    0)�>)  H#c�=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   `
���   `
���      �?!   `
���)@�έw�=2��~��¾�[�=�k���������:              �?        s?y��       �{�	�Z%���A<*�
w
discriminator_loss*a	   �D�>   �D�>      �?!   �D�>) 9l!�f�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   @$ž   @$ž      �?!   @$ž) 駺��=2����ž�XQ�þ�������:              �?        a*X��       �{�	5t^%���AA*�
w
discriminator_loss*a	   �GP?   �GP?      �?!   �GP?)@���W>26�]��?����?�������:              �?        
s
generator_loss*a	    �¾    �¾      �?!    �¾) �G�S��=2�XQ�þ��~��¾�������:              �?        ��c`�       �{�	�Ib%���AF*�
w
discriminator_loss*a	   �7�>   �7�>      �?!   �7�>) ��n!�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   �Ծ   �Ծ      �?!   �Ծ) cg	N�=2��>M|Kվ��~]�[Ӿ�������:              �?        ��k�       �{�	�~f%���AK*�
w
discriminator_loss*a	   �!��>   �!��>      �?!   �!��>) �A�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   ��¾   ��¾      �?!   ��¾)@�de�=2�XQ�þ��~��¾�������:              �?        l<�O�       �{�	�Ej%���AP*�
w
discriminator_loss*a	   �=-�>   �=-�>      �?!   �=-�>) 2n]H��=2��(���>a�Ϭ(�>�������:              �?        
s
generator_loss*a	   @,��   @,��      �?!   @,��)�����=2�*��ڽ�G&�$���������:              �?        �Xs�       �{�	l E'���AU*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>) ��:1(�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) � �B�=25�"�g���0�6�/n���������:              �?        a�L�       �{�	��H'���AZ*�
w
discriminator_loss*a	   `.m�>   `.m�>      �?!   `.m�>)@jbĂ��=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��)@,[i��=2��~��¾�[�=�k���������:              �?        IX�_�       �{�	��L'���A_*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �z��=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   ��ľ   ��ľ      �?!   ��ľ)@�=��=2����ž�XQ�þ�������:              �?        ���Y�       �{�	��P'���Ad*�
w
discriminator_loss*a	    v�>    v�>      �?!    v�>)@�Eoɼ=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   `ľ   `ľ      �?!   `ľ)@fӉCF�=2����ž�XQ�þ�������:              �?        {j�w�       �{�	��T'���Ai*�
w
discriminator_loss*a	   �X��>   �X��>      �?!   �X��>) �9ӳ��=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   @'wӾ   @'wӾ      �?!   @'wӾ) I8�t��=2��>M|Kվ��~]�[Ӿ�������:              �?        �����       �{�	rX'���An*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�'�vY�=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��)����(�j=2��������?�ګ��������:              �?        �k��       �{�	�:\'���As*�
w
discriminator_loss*a	   `pq?   `pq?      �?!   `pq?) �0��,!>26�]��?����?�������:              �?        
s
generator_loss*a	   @�Ⱦ   @�Ⱦ      �?!   @�Ⱦ)����4S�=2;�"�qʾ
�/eq
Ⱦ�������:              �?        �Ks��       �{�	5`'���Ax*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@T]A��=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   �~¾   �~¾      �?!   �~¾)@�U��_�=2�XQ�þ��~��¾�������:              �?        '���       �{�	��S)���A}*�
w
discriminator_loss*a	   �hY�>   �hY�>      �?!   �hY�>)���4��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   �6�Ѿ   �6�Ѿ      �?!   �6�Ѿ)@~�V<5�=2��~]�[Ӿjqs&\�Ѿ�������:              �?        Tpi@�       b�D�	�;Y)���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)@�.b��=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   ��¾   ��¾      �?!   ��¾) �SA��=2�XQ�þ��~��¾�������:              �?        �gx�       b�D�	��])���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)���0L2�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)�p� �=25�"�g���0�6�/n���������:              �?        ��R�       b�D�	��b)���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) Ұ����=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) � Y��w=2�u`P+d����n������������:              �?        /��       b�D�	e�g)���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) a���M�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   ��˾   ��˾      �?!   ��˾) ooȧ=2['�?�;;�"�qʾ�������:              �?        7��       b�D�		l)���A�*�
w
discriminator_loss*a	    (�>    (�>      �?!    (�>) H�p�ƣ=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��)@ t`�K�=2��~��¾�[�=�k���������:              �?        B�C�       b�D�	�p)���A�*�
w
discriminator_loss*a	   �0��>   �0��>      �?!   �0��>) �.���=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	    �þ    �þ      �?!    �þ)@@���*�=2�XQ�þ��~��¾�������:              �?        )C���       b�D�	�t)���A�*�
w
discriminator_loss*a	   �k�>   �k�>      �?!   �k�>) _?�3�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   �&ξ   �&ξ      �?!   �&ξ) ٳ-�g�=2K+�E��Ͼ['�?�;�������:              �?        >9��       b�D�	Ąm+���A�*�
w
discriminator_loss*a	   �
��>   �
��>      �?!   �
��>) rs1�y�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   �\��   �\��      �?!   �\��) �Iu"�=2�*��ڽ�G&�$���������:              �?        ��-��       b�D�	�q+���A�*�
w
discriminator_loss*a	   �uS�>   �uS�>      �?!   �uS�>) �n��=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   ��þ   ��þ      �?!   ��þ) D�A &�=2�XQ�þ��~��¾�������:              �?        p���       b�D�	��u+���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �}��=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   `TϾ   `TϾ      �?!   `TϾ) �
:̫�=2K+�E��Ͼ['�?�;�������:              �?        ��?$�       b�D�	o�y+���A�*�
w
discriminator_loss*a	   @
��>   @
��>      �?!   @
��>)�Hk��:�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)����₁=25�"�g���0�6�/n���������:              �?        �y%l�       b�D�	��}+���A�*�
w
discriminator_loss*a	    ~��>    ~��>      �?!    ~��>) @�/��=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   �&�Ͼ   �&�Ͼ      �?!   �&�Ͼ) �^��p�=2K+�E��Ͼ['�?�;�������:              �?        �ss��       b�D�	k��+���A�*�
w
discriminator_loss*a	    	��>    	��>      �?!    	��>) ���>2>�?�s��>�FF�G ?�������:              �?        
s
generator_loss*a	   �趾   �趾      �?!   �趾) RAN�e�=25�"�g���0�6�/n���������:              �?        M}���       b�D�	х+���A�*�
w
discriminator_loss*a	   @	��>   @	��>      �?!   @	��>)��
Y��=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	    	���    	���      �?!    	���) ���J9�=2�[�=�k���*��ڽ��������:              �?        lv��       b�D�	��+���A�*�
w
discriminator_loss*a	   @	��>   @	��>      �?!   @	��>)��:�y��=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	    >¾    >¾      �?!    >¾) @��}̔=2�XQ�þ��~��¾�������:              �?        ��Sg�       b�D�	d�-���A�*�
w
discriminator_loss*a	   �Io�>   �Io�>      �?!   �Io�>)��M�q��=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	     ¾     ¾      �?!     ¾) @�@�=2��~��¾�[�=�k���������:              �?        �T�4�       b�D�	S�-���A�*�
w
discriminator_loss*a	   �*D�>   �*D�>      �?!   �*D�>)@�✍��=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �����w=2�u`P+d����n������������:              �?        }u6R�       b�D�	P*�-���A�*�
w
discriminator_loss*a	   @t�>   @t�>      �?!   @t�>)�P��-�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   �h�ؾ   �h�ؾ      �?!   �h�ؾ)��B*k��=2�iD*L�پ�_�T�l׾�������:              �?        � ���       b�D�	<�-���A�*�
w
discriminator_loss*a	   @3r�>   @3r�>      �?!   @3r�>)�zm��=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �p��   �p��      �?!   �p��) ��� �=2��~��¾�[�=�k���������:              �?        �r�t�       b�D�	*o�-���A�*�
w
discriminator_loss*a	   �mC�>   �mC�>      �?!   �mC�>)�h�2��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   `�¾   `�¾      �?!   `�¾)@bd���=2�XQ�þ��~��¾�������:              �?        �L��       b�D�	ΰ-���A�*�
w
discriminator_loss*a	   @ |�>   @ |�>      �?!   @ |�>)�����=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) ���p=2豪}0ڰ�������������:              �?        �J���       b�D�	���-���A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>)����P�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   �N��   �N��      �?!   �N��) �����=2�[�=�k���*��ڽ��������:              �?        }ZK��       b�D�	�E�-���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) ��'��=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	    �þ    �þ      �?!    �þ)  `��=2����ž�XQ�þ�������:              �?        �~&$�       b�D�	���/���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) )�P���=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   �p��   �p��      �?!   �p��) D�'�=2��~��¾�[�=�k���������:              �?        �`��       b�D�	_W�/���A�*�
w
discriminator_loss*a	   @|�>   @|�>      �?!   @|�>)���K��=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	    	t��    	t��      �?!    	t��) ���	�=2��~��¾�[�=�k���������:              �?        �^�h�       b�D�	�%�/���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ���|��=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   `fʾ   `fʾ      �?!   `fʾ) �.q#ǥ=2;�"�qʾ
�/eq
Ⱦ�������:              �?        ��g�       b�D�	�X�/���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@T��i�=2})�l a�>pz�w�7�>�������:              �?        
s
generator_loss*a	   `5�ʾ   `5�ʾ      �?!   `5�ʾ) �����=2['�?�;;�"�qʾ�������:              �?        ���F�       b�D�	�2�/���A�*�
w
discriminator_loss*a	    
8�>    
8�>      �?!    
8�>) @��=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   �P��   �P��      �?!   �P��) y����p=2豪}0ڰ�������������:              �?        F� �       b�D�	���/���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �%[K��=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   �^ľ   �^ľ      �?!   �^ľ)@:5N_�=2����ž�XQ�þ�������:              �?        %e�b�       b�D�	n��/���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)��s���=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   �,��   �,��      �?!   �,��) ��>n�=2��~��¾�[�=�k���������:              �?        �R��       b�D�	���/���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�x��C�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) d�gT�=2��~��¾�[�=�k���������:              �?        ��F��       b�D�	X��1���A�*�
w
discriminator_loss*a	   `J�>   `J�>      �?!   `J�>)@���%A�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	    	��    	��      �?!    	��) ���i>�=2�[�=�k���*��ڽ��������:              �?        9$��       b�D�	���1���A�*�
w
discriminator_loss*a	   @ �>   @ �>      �?!   @ �>)�x�TDT�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   `��   `��      �?!   `��)@�Y�I8�=2�[�=�k���*��ڽ��������:              �?        n��4�       b�D�	���1���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)�p�ᗭ�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   �ా   �ా      �?!   �ా) d FE�q=2��n�����豪}0ڰ��������:              �?        ��ޅ�       b�D�	�u�1���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>)@M`g7�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   @
þ   @
þ      �?!   @
þ) �Fb��=2�XQ�þ��~��¾�������:              �?        �"7��       b�D�	Y@�1���A�*�
w
discriminator_loss*a	   ��`�>   ��`�>      �?!   ��`�>)@�����=2�f����>��(���>�������:              �?        
s
generator_loss*a	    \��    \��      �?!    \��) ��5�Ւ=2��~��¾�[�=�k���������:              �?        ~yvD�       b�D�	 �1���A�*�
w
discriminator_loss*a	   �1,�>   �1,�>      �?!   �1,�>) �l��ͨ=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) .�7O�=2G&�$��5�"�g����������:              �?        �W�m�       b�D�	p��1���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)���=�-�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   @��   @��      �?!   @��) Q It=2��n�����豪}0ڰ��������:              �?        �_��       b�D�	��1���A�*�
w
discriminator_loss*a	   �D�>   �D�>      �?!   �D�>) d@�&�t=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   �D��   �D��      �?!   �D��) d@e��r=2��n�����豪}0ڰ��������:              �?        ����       b�D�	���3���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) X�(H}�=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   �hƾ   �hƾ      �?!   �hƾ) ����`�=2
�/eq
Ⱦ����ž�������:              �?        �����       b�D�	���3���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)��ą�h�=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   @Ⱦ   @Ⱦ      �?!   @Ⱦ)�x/k#�=2
�/eq
Ⱦ����ž�������:              �?        ��p,�       b�D�	M��3���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) a8�QW�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   �`��   �`��      �?!   �`��) vB�.��=2G&�$��5�"�g����������:              �?        ��}�       b�D�	���3���A�*�
w
discriminator_loss*a	   `_��>   `_��>      �?!   `_��>) CR.��=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   �]�ؾ   �]�ؾ      �?!   �]�ؾ) d��>�=2�iD*L�پ�_�T�l׾�������:              �?        �;��       b�D�	���3���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) AK����=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   �$ܿ�   �$ܿ�      �?!   �$ܿ�) ~2�q��=2�[�=�k���*��ڽ��������:              �?        �㎤�       b�D�	��3���A�*�
w
discriminator_loss*a	   �R��>   �R��>      �?!   �R��>) W�����=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	    G�˾    G�˾      �?!    G�˾) ��z��=2['�?�;;�"�qʾ�������:              �?        �&��       b�D�	���3���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@ح�r�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   @��   @��      �?!   @��) IS�"$�=2�[�=�k���*��ڽ��������:              �?        	�v��       b�D�	���3���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ·]��=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)@V��$2y=2�u`P+d����n������������:              �?        �\Q�       b�D�	f6���A�*�
w
discriminator_loss*a	   @V�>   @V�>      �?!   @V�>)�FjO¬=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   @.ž   @.ž      �?!   @.ž) �b	�=2����ž�XQ�þ�������:              �?        R�v�       b�D�	]d6���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �)�?�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   ��̾   ��̾      �?!   ��̾)��I�Si�=2['�?�;;�"�qʾ�������:              �?        
/���       b�D�	�T6���A�*�
w
discriminator_loss*a	   @dU�>   @dU�>      �?!   @dU�>)��V��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) RtU�V�=2�[�=�k���*��ڽ��������:              �?        h����       b�D�	Za6���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �)y���=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   ��Ǿ   ��Ǿ      �?!   ��Ǿ) ������=2
�/eq
Ⱦ����ž�������:              �?        �AH�       b�D�	pG6���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) IQR�=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) DAjKmu=2��n�����豪}0ڰ��������:              �?        5�F��       b�D�	,O6���A�*�
w
discriminator_loss*a	    !j�>    !j�>      �?!    !j�>) J�����=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	    þ    þ      �?!    þ) @�����=2�XQ�þ��~��¾�������:              �?        �3�       b�D�	�U#6���A�*�
w
discriminator_loss*a	   @`�>   @`�>      �?!   @`�>)�(��#)i=2���?�ګ>����>�������:              �?        
s
generator_loss*a	     ��     ��      �?!     ��)  �BPY=2���]������|�~���������:              �?        �a��       b�D�	�I'6���A�*�
w
discriminator_loss*a	   @#��>   @#��>      �?!   @#��>)��Ƨ���=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ��*�=2�[�=�k���*��ڽ��������:              �?        �L�D�       b�D�	��G8���A�*�
w
discriminator_loss*a	   �2�>   �2�>      �?!   �2�>)����WР=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)@�"�f�=2��~��¾�[�=�k���������:              �?        ���       b�D�	?_L8���A�*�
w
discriminator_loss*a	   `b�>   `b�>      �?!   `b�>)@J7���=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   `�Ⱦ   `�Ⱦ      �?!   `�Ⱦ) )� t=�=2;�"�qʾ
�/eq
Ⱦ�������:              �?        �XD��       b�D�	?P8���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  �g�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   @p��   @p��      �?!   @p��) I�� ?u=2��n�����豪}0ڰ��������:              �?        ����       b�D�	T8���A�*�
w
discriminator_loss*a	   ��d�>   ��d�>      �?!   ��d�>) ��Fo&�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@2�_}v=2�u`P+d����n������������:              �?        �4�       b�D�	��W8���A�*�
w
discriminator_loss*a	   @3��>   @3��>      �?!   @3��>) )TS"5�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �q���=2G&�$��5�"�g����������:              �?        �ޚx�       b�D�	T�[8���A�*�
w
discriminator_loss*a	   �6�>   �6�>      �?!   �6�>) ���؇�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)@�P%t=2��n�����豪}0ڰ��������:              �?        F("��       b�D�	9�_8���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �	���=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)��@��Af=2���?�ګ�;9��R���������:              �?        QiR��       b�D�	��c8���A�*�
w
discriminator_loss*a	   @̽>   @̽>      �?!   @̽>)��ss龋=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   �d��   �d��      �?!   �d��) �D���y=2�u`P+d����n������������:              �?        Fx�
�       b�D�	�՜:���A�*�
w
discriminator_loss*a	   �l�>   �l�>      �?!   �l�>) ���k��=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   `
0��   `
0��      �?!   `
0��) ]#GZԈ=2�*��ڽ�G&�$���������:              �?        a����       b�D�	8�:���A�*�
w
discriminator_loss*a	   ` ��>   ` ��>      �?!   ` ��>) ��PRة=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   @n¾   @n¾      �?!   @n¾) ��ϩ:�=2�XQ�þ��~��¾�������:              �?        VTF�       b�D�	��:���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@Xʥ�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   �ܽ�   �ܽ�      �?!   �ܽ�) �ٺ�܋=2�[�=�k���*��ڽ��������:              �?        �u&�       b�D�	1��:���A�*�
w
discriminator_loss*a	    8��>    8��>      �?!    8��>)  ��>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	    `��    `��      �?!    `��)@,�AQvw=2�u`P+d����n������������:              �?        W_��       b�D�	�z�:���A�*�
w
discriminator_loss*a	   �T�>   �T�>      �?!   �T�>) �D�MY�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   @н�   @н�      �?!   @н�)�XF�bƋ=2�*��ڽ�G&�$���������:              �?        [�k*�       b�D�	;��:���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) H�L���=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@��mxas=2��n�����豪}0ڰ��������:              �?        t�_�       b�D�	�ֵ:���A�*�
w
discriminator_loss*a	   �f�>   �f�>      �?!   �f�>)@(�~؄�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@���!t�=2��~��¾�[�=�k���������:              �?        "RT�       b�D�	o��:���A�*�
w
discriminator_loss*a	   @ϒ�>   @ϒ�>      �?!   @ϒ�>) ����=2�f����>��(���>�������:              �?        
s
generator_loss*a	   `˾   `˾      �?!   `˾) �&ӛ�=2['�?�;;�"�qʾ�������:              �?        ç��       b�D�	I��<���A�*�
w
discriminator_loss*a	   �P�>   �P�>      �?!   �P�>)���B��=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�T@5�0m=2��������?�ګ��������:              �?        hh�       b�D�	��<���A�*�
w
discriminator_loss*a	   �	 �>   �	 �>      �?!   �	 �>) ����=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) AD<��=25�"�g���0�6�/n���������:              �?        V�?�       b�D�	'��<���A�*�
w
discriminator_loss*a	   �&�>   �&�>      �?!   �&�>) ���=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) ���=2��~��¾�[�=�k���������:              �?        ߫ˣ�       b�D�	]�<���A�*�
w
discriminator_loss*a	   �=�>   �=�>      �?!   �=�>) A;�{�=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   �	$��   �	$��      �?!   �	$��) �b�y��=2�*��ڽ�G&�$���������:              �?        ����       b�D�	�_�<���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �)5_��=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@*@��_=2�5�L�����]�����������:              �?        O�b�       b�D�	#��<���A�*�
w
discriminator_loss*a	   �h�>   �h�>      �?!   �h�>) ����w=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@��GFq=2豪}0ڰ�������������:              �?        <�^�       b�D�	~� =���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)@^��҇�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	    ķ�    ķ�      �?!    ķ�)  b|��=25�"�g���0�6�/n���������:              �?        ���D�       b�D�	{�=���A�*�
w
discriminator_loss*a	   �L��>   �L��>      �?!   �L��>) {�C��=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	    �ƾ    �ƾ      �?!    �ƾ) ��੟=2
�/eq
Ⱦ����ž�������:              �?        '�1�       b�D�	��A?���A�*�
w
discriminator_loss*a	   � �>   � �>      �?!   � �>) D�\P�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	    	ȶ�    	ȶ�      �?!    	ȶ�) ����7�=25�"�g���0�6�/n���������:              �?        �3( �       b�D�	JF?���A�*�
w
discriminator_loss*a	   �P�>   �P�>      �?!   �P�>) ����x�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	    
P��    
P��      �?!    
P��)  2�x�=25�"�g���0�6�/n���������:              �?        �Z`��       b�D�	�xJ?���A�*�
w
discriminator_loss*a	   �	(�>   �	(�>      �?!   �	(�>) ��@<�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   �8��   �8��      �?!   �8��) ����~=20�6�/n���u`P+d���������:              �?        a����       b�D�	�hN?���A�*�
w
discriminator_loss*a	    +��>    +��>      �?!    +��>) �"m�=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	    \��    \��      �?!    \��)  �p`d�=2�*��ڽ�G&�$���������:              �?        |�V��       b�D�	KR?���A�*�
w
discriminator_loss*a	   �
��>   �
��>      �?!   �
��>)��#��?�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   `X��   `X��      �?!   `X��)@2!��bw=2�u`P+d����n������������:              �?        *˃��       b�D�	�V?���A�*�
w
discriminator_loss*a	   �,�>   �,�>      �?!   �,�>)@`1N��=2�f����>��(���>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)��`��|�=25�"�g���0�6�/n���������:              �?        �v�+�       b�D�	,[?���A�*�
w
discriminator_loss*a	    -��>    -��>      �?!    -��>) ����=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) ai�R_�=2��~��¾�[�=�k���������:              �?        ��u��       b�D�	Q`?���A�*�
w
discriminator_loss*a	    ;��>    ;��>      �?!    ;��>) ��m��=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   @9Q�   @9Q�      �?!   @9Q�) �t�5��=2�uE���⾮��%��������:              �?        � ��       b�D�	Mu�A���A�*�
w
discriminator_loss*a	   �
��>   �
��>      �?!   �
��>)@/��Б=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�(���{c=2;9��R���5�L���������:              �?        @d��       b�D�	
6�A���A�*�
w
discriminator_loss*a	   �(�>   �(�>      �?!   �(�>) Bs�H��=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	    "�ľ    "�ľ      �?!    "�ľ) @H0X�=2����ž�XQ�þ�������:              �?        XB�o�       b�D�	���A���A�*�
w
discriminator_loss*a	    )>�>    )>�>      �?!    )>�>) ����=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	    )"ƾ    )"ƾ      �?!    )"ƾ) ��9��=2
�/eq
Ⱦ����ž�������:              �?        ��x�       b�D�	�3�A���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)���5�۩=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   �	Ƚ�   �	Ƚ�      �?!   �	Ƚ�)���%t��=2�*��ڽ�G&�$���������:              �?        �{M"�       b�D�	��A���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@��Q� �=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	    �þ    �þ      �?!    �þ) ��˗C�=2�XQ�þ��~��¾�������:              �?        zN��       b�D�	��A���A�*�
w
discriminator_loss*a	   �0�>   �0�>      �?!   �0�>) )�ı�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   @
H��   @
H��      �?!   @
H��) �F�X��=2�[�=�k���*��ڽ��������:              �?        Hͧ�       b�D�	�A�A���A�*�
w
discriminator_loss*a	   @p�>   @p�>      �?!   @p�>) ����w=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   @ਾ   @ਾ      �?!   @ਾ)�܀)(Vc=2;9��R���5�L���������:              �?        c���       b�D�	L�A���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)����zb�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   @ļ�   @ļ�      �?!   @ļ�)�h�O�ۉ=2�*��ڽ�G&�$���������:              �?        �<�       b�D�	�D���A�*�
w
discriminator_loss*a	    A4�>    A4�>      �?!    A4�>) �'M��=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) � ��9x=2�u`P+d����n������������:              �?        D�A��       b�D�	�D���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) � �x=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	    h��    h��      �?!    h��) �P�[�l=2��������?�ګ��������:              �?        Lr���       b�D�	�D���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)�̶��=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   @	���   @	���      �?!   @	���) Y�@�,�=2��~��¾�[�=�k���������:              �?        *m��       b�D�	��!D���A�*�
w
discriminator_loss*a	   @C�>   @C�>      �?!   @C�>) ��Qi0�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   @�¾   @�¾      �?!   @�¾) �"\r��=2�XQ�þ��~��¾�������:              �?        �_q��       b�D�	�%D���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ەO�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   �ȵ�   �ȵ�      �?!   �ȵ�) ���զ}=20�6�/n���u`P+d���������:              �?        �x���       b�D�	G~)D���A�*�
w
discriminator_loss*a	    x�>    x�>      �?!    x�>) �lε�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   �`��   �`��      �?!   �`��) � �(�l=2��������?�ګ��������:              �?        <�;��       b�D�	�I-D���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�PD�ڞ�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) N ��Ya=2;9��R���5�L���������:              �?        R���       b�D�	�1D���A�*�
w
discriminator_loss*a	   �p�>   �p�>      �?!   �p�>) �iѡ*�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   @�ž   @�ž      �?!   @�ž) �r"�ݝ=2
�/eq
Ⱦ����ž�������:              �?        t����       b�D�	ih�F���A�*�
w
discriminator_loss*a	   �v�>   �v�>      �?!   �v�>) �j$��=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)@VA�v=2�u`P+d����n������������:              �?        7����       b�D�	�:�F���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) >��է=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   @F��   @F��      �?!   @F��) �����=2��~��¾�[�=�k���������:              �?        �0Х�       b�D�	��F���A�*�
w
discriminator_loss*a	    z�>    z�>      �?!    z�>)@L��
Ԝ=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   �<��   �<��      �?!   �<��)@V����t=2��n�����豪}0ڰ��������:              �?        ��       b�D�	�F���A�*�
w
discriminator_loss*a	    	��>    	��>      �?!    	��>) �B�w�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   @ا�   @ا�      �?!   @ا�)�(`Z5�a=2;9��R���5�L���������:              �?        Rg ��       b�D�	^��F���A�*�
w
discriminator_loss*a	    @�>    @�>      �?!    @�>) ܲ��=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   ��   ��      �?!   ��) �Y�e��=2a�Ϭ(���(����������:              �?        �����       b�D�	�=�F���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) A�)��=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   @��   @��      �?!   @��) ��zl~=20�6�/n���u`P+d���������:              �?        !̐j�       b�D�	H!�F���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�ȶ�R�=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ���)B�=2��~��¾�[�=�k���������:              �?        u���       b�D�	n�F���A�*�
w
discriminator_loss*a	   @!��>   @!��>      �?!   @!��>)���@�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) �DT��{=20�6�/n���u`P+d���������:              �?        ����       b�D�	��I���A�*�
w
discriminator_loss*a	   @4�>   @4�>      �?!   @4�>) q���h�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��)�PD�\Ԉ=2�*��ڽ�G&�$���������:              �?        �� ��       b�D�	ȐI���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@���ޚ=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   �4��   �4��      �?!   �4��) �hDB�=2��~��¾�[�=�k���������:              �?        ��X^�       b�D�	�`I���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)   %d�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   ��   ��      �?!   ��) � m�m=2豪}0ڰ�������������:              �?        BY��       b�D�	'I���A�*�
w
discriminator_loss*a	   �E&�>   �E&�>      �?!   �E&�>) }����=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@r��O�=2��~��¾�[�=�k���������:              �?        �K�:�       b�D�	5�I���A�*�
w
discriminator_loss*a	   �]3�>   �]3�>      �?!   �]3�>) ��^M�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �
(��   �
(��      �?!   �
(��) ��\Ő�=2�*��ڽ�G&�$���������:              �?        2,T�       b�D�	9 I���A�*�
w
discriminator_loss*a	   �f�>   �f�>      �?!   �f�>) d�>[�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) @!$!`=2�5�L�����]�����������:              �?        �jV�       b�D�	;�I���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) 6�|�m�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	    8��    8��      �?!    8��)@��;��~=20�6�/n���u`P+d���������:              �?        s���       b�D�	 �I���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)@��]�ě=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   @!��   @!��      �?!   @!��)��b�H'�=2�[�=�k���*��ڽ��������:              �?        �+��       b�D�	�jwK���A�*�
w
discriminator_loss*a	   �ܷ>   �ܷ>      �?!   �ܷ>) Ra12ʁ=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   �̰�   �̰�      �?!   �̰�) i!�2�q=2豪}0ڰ�������������:              �?        ����       b�D�	�|K���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ��Ym۳=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	    4��    4��      �?!    4��) HZMm�=2�[�=�k���*��ڽ��������:              �?        �����       b�D�	�l�K���A�*�
w
discriminator_loss*a	    	Ĺ>    	Ĺ>      �?!    	Ĺ>) �B����=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) B"���=25�"�g���0�6�/n���������:              �?        r���       b�D�	��K���A�*�
w
discriminator_loss*a	   �2�>   �2�>      �?!   �2�>)@���p�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   �췾   �췾      �?!   �췾)��Q�=25�"�g���0�6�/n���������:              �?        l��       b�D�	W��K���A�*�
w
discriminator_loss*a	    \��>    \��>      �?!    \��>)  �L�=2�f����>��(���>�������:              �?        
s
generator_loss*a	   �l��   �l��      �?!   �l��) �$l��=2��~��¾�[�=�k���������:              �?        ㊁��       b�D�	�v�K���A�*�
w
discriminator_loss*a	   @N��>   @N��>      �?!   @N��>) �V��+�=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	    %о    %о      �?!    %о) ��^\"�=2jqs&\�ѾK+�E��Ͼ�������:              �?        �����       b�D�	e��K���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@��Δҝ=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   @`��   @`��      �?!   @`��)���1�=2G&�$��5�"�g����������:              �?        �!�m�       b�D�	�S�K���A�*�
w
discriminator_loss*a	   �y��>   �y��>      �?!   �y��>) ,���=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) a�"ȑ=2��~��¾�[�=�k���������:              �?        C�l�       b�D�	^�M���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>)@xl�Eɖ=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) H .%�g=2���?�ګ�;9��R���������:              �?        �����       b�D�	�9�M���A�*�
w
discriminator_loss*a	    /i�>    /i�>      �?!    /i�>) j�Φ�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ���ͳ�=2G&�$��5�"�g����������:              �?        �{X-�       b�D�	)�M���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>)  y=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   �`��   �`��      �?!   �`��) � %H�r=2��n�����豪}0ڰ��������:              �?        �}{��       b�D�	�M���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@�ؙ@�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@v���=2��~��¾�[�=�k���������:              �?        �mpO�       b�D�	3�M���A�*�
w
discriminator_loss*a	   @6�>   @6�>      �?!   @6�>) ��5���=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   ��ž   ��ž      �?!   ��ž)@(��ǝ=2����ž�XQ�þ�������:              �?        ���9�       b�D�	�w�M���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) iQ� �=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   @��   @��      �?!   @��)���a�=25�"�g���0�6�/n���������:              �?        %��6�       b�D�	�S�M���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ү.�t�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) ᠅p=2豪}0ڰ�������������:              �?        �:�       b�D�	��N���A�*�
w
discriminator_loss*a	   �@��>   �@��>      �?!   �@��>) �E2��=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �,��   �,��      �?!   �,��) ��~|=20�6�/n���u`P+d���������:              �?        ���{�       b�D�	x�wP���A�*�
w
discriminator_loss*a	   �L�>   �L�>      �?!   �L�>)�lT���=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)� ��1}m=2豪}0ڰ�������������:              �?        |r-��       b�D�	4X|P���A�*�
w
discriminator_loss*a	    I��>    I��>      �?!    I��>) �����=2pz�w�7�>I��P=�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@*`�G�U=2��|�~���MZ��K���������:              �?        p�\�       b�D�	��P���A�*�
w
discriminator_loss*a	    �~�>    �~�>      �?!    �~�>) � PP�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
s
generator_loss*a	   @
���   @
���      �?!   @
���)�HӚ{_�=25�"�g���0�6�/n���������:              �?        ���       b�D�	A��P���A�*�
w
discriminator_loss*a	   @?h�>   @?h�>      �?!   @?h�>)��?F�=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   `3�þ   `3�þ      �?!   `3�þ)@���~a�=2�XQ�þ��~��¾�������:              �?        H���       b�D�	�=�P���A�*�
w
discriminator_loss*a	   �L�>   �L�>      �?!   �L�>) B�����=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	    �þ    �þ      �?!    �þ) ��!���=2�XQ�þ��~��¾�������:              �?        R⾌�       b�D�	n�P���A�*�
w
discriminator_loss*a	   `:�>   `:�>      �?!   `:�>)@v�d~ô=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   �8��   �8��      �?!   �8��) �xz�=2�*��ڽ�G&�$���������:              �?        ����       b�D�	A��P���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)@�J���=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   @$ƾ   @$ƾ      �?!   @$ƾ) !��gV�=2
�/eq
Ⱦ����ž�������:              �?         x��       b�D�	�]�P���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) ��1��{=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) 2��$b=2;9��R���5�L���������:              �?        X��&�       b�D�	=�/S���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) @��ܟ=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	    `��    `��      �?!    `��)@��}J�p=2豪}0ڰ�������������:              �?         �`+�       b�D�	�5S���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�� ��=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) y���s=2��n�����豪}0ڰ��������:              �?        ��K��       b�D�	�u9S���A�*�
w
discriminator_loss*a	    D�>    D�>      �?!    D�>)@,�|/��=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) R�o=2豪}0ڰ�������������:              �?        =4yh�       �N�	��<S���A*�
w
discriminator_loss*a	   @"��>   @"��>      �?!   @"��>)��,�.�=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@B����=2��~��¾�[�=�k���������:              �?        :(
W�       �{�	�ڨY���A*�
w
discriminator_loss*a	   �\�>   �\�>      �?!   �\�>)���L��=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �����z=20�6�/n���u`P+d���������:              �?        Q����       �{�	#��Y���A
*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ȃ7}�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) @,Gs=2��n�����豪}0ڰ��������:              �?        �5���       �{�	t��Y���A*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ���zH�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   �	���   �	���      �?!   �	���)��R4e�=2G&�$��5�"�g����������:              �?        fn-a�       �{�	⺷Y���A*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  �4I�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   `
��   `
��      �?!   `
��) ]� �E�=2�[�=�k���*��ڽ��������:              �?        @;	I�       �{�	��Y���A*�
w
discriminator_loss*a	   `	h�>   `	h�>      �?!   `	h�>) ��b7�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   @@��   @@��      �?!   @@��)�� E��h=2��������?�ګ��������:              �?        #�Q��       �{�	�F�Y���A*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �!�c-q=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) ��.T=2��|�~���MZ��K���������:              �?        �6~?�       �{�	~�Y���A#*�
w
discriminator_loss*a	   �ķ>   �ķ>      �?!   �ķ>) xxAv��=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   `ܱ�   `ܱ�      �?!   `ܱ�)@ZPMV�s=2��n�����豪}0ڰ��������:              �?        �k���       �{�	��Y���A(*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ��>-�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   �D��   �D��      �?!   �D��)�q8�f�=25�"�g���0�6�/n���������:              �?        if$z�       �{�	�\W\���A-*�
w
discriminator_loss*a	   � �>   � �>      �?!   � �>) d ��v=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   �`��   �`��      �?!   �`��) B@a$�b=2;9��R���5�L���������:              �?        <��&�       �{�	WB[\���A2*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �~��=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   ��ľ   ��ľ      �?!   ��ľ) $��mN�=2����ž�XQ�þ�������:              �?        [�]��       �{�	�_\���A7*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)���pѢ=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   �r¾   �r¾      �?!   �r¾) ����C�=2�XQ�þ��~��¾�������:              �?        �kM��       �{�	��b\���A<*�
w
discriminator_loss*a	    ,�>    ,�>      �?!    ,�>) ����ny=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��) ��'��v=2�u`P+d����n������������:              �?        |����       �{�	��f\���AA*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@��?ݘ=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) 2@uFpi=2��������?�ګ��������:              �?        }P�       �{�	�j\���AF*�
w
discriminator_loss*a	   � ��>   � ��>      �?!   � ��>) 	#,ʒ�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) x��b=2;9��R���5�L���������:              �?        �xcQ�       �{�	^n\���AK*�
w
discriminator_loss*a	   �T�>   �T�>      �?!   �T�>)@��(=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)�p`�w�b=2;9��R���5�L���������:              �?        � ���       �{�	�7r\���AP*�
w
discriminator_loss*a	    �p�>    �p�>      �?!    �p�>) L�ՒF�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   �~�Ҿ   �~�Ҿ      �?!   �~�Ҿ)@�󑅵=2��~]�[Ӿjqs&\�Ѿ�������:              �?        �K���       �{�	2��^���AU*�
w
discriminator_loss*a	    �
�>    �
�>      �?!    �
�>)@��X�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   �Զ�   �Զ�      �?!   �Զ�)�d��I�=25�"�g���0�6�/n���������:              �?        ⽤o�       �{�	��^���AZ*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  ���w=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) �pR	b=2;9��R���5�L���������:              �?        [�W��       �{�	?��^���A_*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) �ٸ���=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  R�Nq=2豪}0ڰ�������������:              �?        �ZI��       �{�	�^�^���Ad*�
w
discriminator_loss*a	   �@�>   �@�>      �?!   �@�>) ��ŭ4�=2;�"�q�>['�?��>�������:              �?        
s
generator_loss*a	   ��ƾ   ��ƾ      �?!   ��ƾ) 쬞�v�=2
�/eq
Ⱦ����ž�������:              �?        ��o�       �{�	�*�^���Ai*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)�P�g��=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   �
@��   �
@��      �?!   �
@��) 9�)w=2�u`P+d����n������������:              �?        ���&�       �{�	���^���An*�
w
discriminator_loss*a	   @H�>   @H�>      �?!   @H�>)�� /(��=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   @��   @��      �?!   @��) ��� p=2豪}0ڰ�������������:              �?        ����       �{�	���^���As*�
w
discriminator_loss*a	   �/?�>   �/?�>      �?!   �/?�>)@�
'�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   @vþ   @vþ      �?!   @vþ) �N`ܫ�=2�XQ�þ��~��¾�������:              �?        �W���       �{�	��^���Ax*�
w
discriminator_loss*a	   � �>   � �>      �?!   � �>) 1 ��[=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) $  P=2�MZ��K���u��gr���������:              �?        �:<�       �{�	.�Fa���A}*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) �S�R��=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) H����b=2;9��R���5�L���������:              �?        oB��       b�D�	Z�Ja���A�*�
w
discriminator_loss*a	    .�>    .�>      �?!    .�>)@L/.��=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �����=25�"�g���0�6�/n���������:              �?        �7Ee�       b�D�	d�Na���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) L_�Q�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	    ȷ�    ȷ�      �?!    ȷ�) ȍ6���=25�"�g���0�6�/n���������:              �?        �p�6�       b�D�	�WRa���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �O�ț�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	    ྾    ྾      �?!    ྾) \�!Zʍ=2�[�=�k���*��ڽ��������:              �?        '���       b�D�	�2Va���A�*�
w
discriminator_loss*a	   @ж>   @ж>      �?!   @ж>)��AVRC�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   `	`��   `	`��      �?!   `	`��) �B61�j=2��������?�ګ��������:              �?        
i�       b�D�	Za���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) 1���q=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   �૾   �૾      �?!   �૾)�<��$Hh=2��������?�ګ��������:              �?        �2�V�       b�D�	 %^a���A�*�
w
discriminator_loss*a	   @4H�>   @4H�>      �?!   @4H�>)�Pu'���=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	    3�˾    3�˾      �?!    3�˾) ��k�3�=2['�?�;;�"�qʾ�������:              �?        ��X/�       b�D�	�4ba���A�*�
w
discriminator_loss*a	   `{��>   `{��>      �?!   `{��>) ��Z��=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@�s���=2��~��¾�[�=�k���������:              �?        �r��       b�D�	Yf�c���A�*�
w
discriminator_loss*a	   �,�>   �,�>      �?!   �,�>)@�rd�X�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   @P��   @P��      �?!   @P��) ��p��t=2��n�����豪}0ڰ��������:              �?        Bw��       b�D�	wh�c���A�*�
w
discriminator_loss*a	   �h��>   �h��>      �?!   �h��>) B��	�=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   �>p��   �>p��      �?!   �>p��)��K�=2�[�=�k���*��ڽ��������:              �?        �ۉM�       b�D�	�X�c���A�*�
w
discriminator_loss*a	   �	��>   �	��>      �?!   �	��>) �E���=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   �	�   �	�      �?!   �	�) ��(f{=20�6�/n���u`P+d���������:              �?        Z9���       b�D�	��c���A�*�
w
discriminator_loss*a	   � �>   � �>      �?!   � �>)��� �=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	    H��    H��      �?!    H��)@x,Wk=20�6�/n���u`P+d���������:              �?        ��H#�       b�D�	p�c���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �&�^2�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	    D��    D��      �?!    D��) <���=2G&�$��5�"�g����������:              �?        �+��       b�D�	&��c���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) ����,�=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   �p��   �p��      �?!   �p��) � �w_=2�5�L�����]�����������:              �?        oX��       b�D�	z��c���A�*�
w
discriminator_loss*a	   �E
�>   �E
�>      �?!   �E
�>)@(��l��=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   @AH��   @AH��      �?!   @AH��)�%����=2�[�=�k���*��ڽ��������:              �?        �]Kb�       b�D�	S��c���A�*�
w
discriminator_loss*a	    .~�>    .~�>      �?!    .~�>)@�� �=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	    `��    `��      �?!    `��) z<Ռ=2�[�=�k���*��ڽ��������:              �?        ����       b�D�	��f���A�*�
w
discriminator_loss*a	    |�>    |�>      �?!    |�>) �C�K�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   �8��   �8��      �?!   �8��) �5k�l=2��������?�ګ��������:              �?        �A���       b�D�	�܌f���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) DAK$2y=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   �H��   �H��      �?!   �H��)@��IK�Y=2���]������|�~���������:              �?        �>M��       b�D�	2ϐf���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) � �'n=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   �  ��   �  ��      �?!   �  ��) 	 � P=2�MZ��K���u��gr���������:              �?        ��V�       b�D�	���f���A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>) I�e4v�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   �
�   �
�      �?!   �
�) �c��=25�"�g���0�6�/n���������:              �?        ���(�       b�D�	��f���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �C�p��=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	    	���    	���      �?!    	���) �F�}=20�6�/n���u`P+d���������:              �?        �4G��       b�D�	�2�f���A�*�
w
discriminator_loss*a	   @'V�>   @'V�>      �?!   @'V�>)�$(�q¬=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	    '�̾    '�̾      �?!    '�̾) �O}�
�=2['�?�;;�"�qʾ�������:              �?        �'�+�       b�D�	��f���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) �x���=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   �
���   �
���      �?!   �
���)@g-!*~=20�6�/n���u`P+d���������:              �?        �u�.�       b�D�	9ۤf���A�*�
w
discriminator_loss*a	   �D�>   �D�>      �?!   �D�>)@|���K�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   �ಾ   �ಾ      �?!   �ಾ) D�JDv=2�u`P+d����n������������:              �?        �`1��       b�D�	K�Vi���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) �0�Ml=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   �H��   �H��      �?!   �H��) $�G�P=2�MZ��K���u��gr���������:              �?        v
v�       b�D�	��Zi���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) b���=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   @0��   @0��      �?!   @0��)�(�BK�`=2;9��R���5�L���������:              �?        �(�%�       b�D�	�^i���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) 6�Vn�>2��Zr[v�>O�ʗ��>�������:              �?        
s
generator_loss*a	    ৾    ৾      �?!    ৾)  �(�a=2;9��R���5�L���������:              �?        U�D��       b�D�	E{bi���A�*�
w
discriminator_loss*a	   �
�>   �
�>      �?!   �
�>) ����e�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	    `��    `��      �?!    `��)   �#�N=2�u��gr��R%�������������:              �?        �5	��       b�D�	bEfi���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ���h�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	    `��    `��      �?!    `��)@0��^�y=2�u`P+d����n������������:              �?        ����       b�D�	�Wji���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ����|=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) � � l=2��������?�ګ��������:              �?        P���       b�D�	�Pni���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) $u-e=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) 	��P=2�MZ��K���u��gr���������:              �?        L$��       b�D�	x*ri���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) x����g=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)�p��%�b=2;9��R���5�L���������:              �?        ���	�       b�D�	6�l���A�*�
w
discriminator_loss*a	   @	h�>   @	h�>      �?!   @	h�>) YE��Ґ=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��) b�JMHb=2;9��R���5�L���������:              �?        M^2�       b�D�	��l���A�*�
w
discriminator_loss*a	    ȹ>    ȹ>      �?!    ȹ>) H��vń=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   �
ா   �
ா      �?!   �
ா) �À4�m=2豪}0ڰ�������������:              �?        [�g�       b�D�	��	l���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) !P��=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)�d:o=2豪}0ڰ�������������:              �?        ǳ��       b�D�	ڍl���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �T���=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �^��=2�[�=�k���*��ڽ��������:              �?        y����       b�D�	&_l���A�*�
w
discriminator_loss*a	   �{w�>   �{w�>      �?!   �{w�>)��B����=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �X��   �X��      �?!   �X��) B��p=2豪}0ڰ�������������:              �?        k���       b�D�	=l���A�*�
w
discriminator_loss*a	   �l�>   �l�>      �?!   �l�>) "m��$�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) 1�Շ5S=2��|�~���MZ��K���������:              �?        �l���       b�D�	�l���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  ���=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   �@��   �@��      �?!   �@��) 1 ��R=2�MZ��K���u��gr���������:              �?        �wx��       b�D�	l���A�*�
w
discriminator_loss*a	   �	�>   �	�>      �?!   �	�>) �"���=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   @خ�   @خ�      �?!   @خ�)� b���m=2豪}0ڰ�������������:              �?        qB�       b�D�	�`�n���A�*�
w
discriminator_loss*a	   `A"�>   `A"�>      �?!   `A"�>) �q);��=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   �$ĺ�   �$ĺ�      �?!   �$ĺ�) ���mc�=2G&�$��5�"�g����������:              �?        |g�e�       b�D�	�q�n���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ����=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) BI0Cr=2��n�����豪}0ڰ��������:              �?        4J��       b�D�	��n���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@�^�(�=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   @X��   @X��      �?!   @X��) !���3_=2�5�L�����]�����������:              �?        !��<�       b�D�	���n���A�*�
w
discriminator_loss*a	   @
,�>   @
,�>      �?!   @
,�>) �f���=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   �8��   �8��      �?!   �8��) �DM��~=20�6�/n���u`P+d���������:              �?        ����       b�D�	#��n���A�*�
w
discriminator_loss*a	   �h�>   �h�>      �?!   �h�>)�<��k=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) 1@e(<Y=2���]������|�~���������:              �?        �����       b�D�	���n���A�*�
w
discriminator_loss*a	   `X�>   `X�>      �?!   `X�>) c�2��=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��)� �H=2R%������39W$:����������:              �?        Ԗ�c�       b�D�	���n���A�*�
w
discriminator_loss*a	    D�>    D�>      �?!    D�>) *��q��=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) �!��oq=2豪}0ڰ�������������:              �?        ��f�       b�D�	k\�n���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ���~�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	    P��    P��      �?!    P��) H �ͣN=2�u��gr��R%�������������:              �?        �ͳn�       b�D�	 ��q���A�*�
w
discriminator_loss*a	    Щ>    Щ>      �?!    Щ>) H �L�d=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)@ [�Q=2�MZ��K���u��gr���������:              �?        ����       b�D�	��q���A�*�
w
discriminator_loss*a	   �(�>   �(�>      �?!   �(�>)��n�Ua�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)��m=2��������?�ګ��������:              �?        �����       b�D�	�V�q���A�*�
w
discriminator_loss*a	    4�>    4�>      �?!    4�>) H�]���=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) 1�aY=2���]������|�~���������:              �?        �,�Y�       b�D�	Ω�q���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�(��˞d=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   �p��   �p��      �?!   �p��) $ u?U=2��|�~���MZ��K���������:              �?        ֍�-�       b�D�	�k�q���A�*�
w
discriminator_loss*a	    |�>    |�>      �?!    |�>) �Y[�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   �x��   �x��      �?!   �x��) Ā���W=2���]������|�~���������:              �?        B6���       b�D�	��q���A�*�
w
discriminator_loss*a	    >J�>    >J�>      �?!    >J�>)@8�Gd�=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@��_��Z=2���]������|�~���������:              �?        b��7�       b�D�	��q���A�*�
w
discriminator_loss*a	   @d�>   @d�>      �?!   @d�>) ��3.��=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   �p��   �p��      �?!   �p��) i`���e=2���?�ګ�;9��R���������:              �?        ��S�       b�D�	���q���A�*�
w
discriminator_loss*a	   �	��>   �	��>      �?!   �	��>)@��䄑=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   �h��   �h��      �?!   �h��)�a��k=2��������?�ګ��������:              �?        ˘z��       b�D�	�ɓt���A�*�
w
discriminator_loss*a	   �X�>   �X�>      �?!   �X�>) "�(���=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��)�p@oM�`=2;9��R���5�L���������:              �?        �����       b�D�	%#�t���A�*�
w
discriminator_loss*a	   �%��>   �%��>      �?!   �%��>) �r1ʥ�=2E��a�W�>�ѩ�-�>�������:              �?        
s
generator_loss*a	   �H��   �H��      �?!   �H��) ���K�p=2豪}0ڰ�������������:              �?        `��       b�D�	O}�t���A�*�
w
discriminator_loss*a	   �
��>   �
��>      �?!   �
��>) 9w�S��=2�XQ��>�����>�������:              �?        
s
generator_loss*a	    L��    L��      �?!    L��) ��I�=2�*��ڽ�G&�$���������:              �?        �M��       b�D�	�Z�t���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�z%ė=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �}��M�=25�"�g���0�6�/n���������:              �?        ����       b�D�	���t���A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>) ���jV=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) � �'HH=2R%������39W$:����������:              �?        �h{~�       b�D�	̰�t���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��X���=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   �蹾   �蹾      �?!   �蹾) �7�+��=2G&�$��5�"�g����������:              �?        ^�+-�       b�D�	ڬt���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�GS=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��)  p E=2R%������39W$:����������:              �?        �&��       b�D�	|J�t���A�*�
w
discriminator_loss*a	   �
$�>   �
$�>      �?!   �
$�>) r�O��=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   `8��   `8��      �?!   `8��) EQrl{e=2���?�ګ�;9��R���������:              �?        ��t��       b�D�	x��w���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)�౭פ�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) �"b�)r=2��n�����豪}0ڰ��������:              �?        ��u�       b�D�	�y�w���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) �C�鳃=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   �h��   �h��      �?!   �h��) ���[7i=2��������?�ګ��������:              �?        �Z�M�       b�D�	B�w���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@n�͕�s=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��) d �`p=2豪}0ڰ�������������:              �?        ����       b�D�	,�w���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) ��h��m=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��) � ��[=2�5�L�����]�����������:              �?        ,Ka��       b�D�	p�w���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) � FGFq=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) � h&�d=2���?�ګ�;9��R���������:              �?        #�Mg�       b�D�	�˫w���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) � �N�=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) � h	�\=2�5�L�����]�����������:              �?        ��o��       b�D�	,��w���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@��źp�=2���%�>�uE����>�������:              �?        
s
generator_loss*a	   �蠾   �蠾      �?!   �蠾) $�+'�Q=2�MZ��K���u��gr���������:              �?        О���       b�D�	h��w���A�*�
w
discriminator_loss*a	   �h�>   �h�>      �?!   �h�>) �ǌ�x�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   ��   ��      �?!   ��) y���q=2��n�����豪}0ڰ��������:              �?        K���       b�D�	��z���A�*�
w
discriminator_loss*a	   �D�>   �D�>      �?!   �D�>)@8��$�p=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   `8��   `8��      �?!   `8��) [�Q��c=2;9��R���5�L���������:              �?        \���       b�D�	
	�z���A�*�
w
discriminator_loss*a	    	Ծ>    	Ծ>      �?!    	Ծ>) �B���=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	    䱾    䱾      �?!    䱾) ��.<t=2��n�����豪}0ڰ��������:              �?        4��<�       b�D�	G4�z���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) )����=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ���|^m=2豪}0ڰ�������������:              �?        �;�%�       b�D�	G�z���A�*�
w
discriminator_loss*a	   � `�>   � `�>      �?!   � `�>)��!A=2X$�z�>.��fc��>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���) 	 �D:=2X$�z��
�}�����������:              �?        zut9�       b�D�	=K�z���A�*�
w
discriminator_loss*a	   �	|�>   �	|�>      �?!   �	|�>) �%��Z�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��)  �ej=2��������?�ګ��������:              �?        ֬��       b�D�	�H�z���A�*�
w
discriminator_loss*a	   @ܰ>   @ܰ>      �?!   @ܰ>) �a��q=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   @��   @��      �?!   @��)��@1'N=2�u��gr��R%�������������:              �?        ����       b�D�	�b�z���A�*�
w
discriminator_loss*a	   �	0�>   �	0�>      �?!   �	0�>)��BYW�c=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	   �@��   �@��      �?!   �@��)� Â�C=239W$:���.��fc����������:              �?        W���       b�D�	m�z���A�*�
w
discriminator_loss*a	   �0�>   �0�>      �?!   �0�>) dq��~=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �	X$s=2��n�����豪}0ڰ��������:              �?        {˴X�       b�D�	V��~���A�*�
w
discriminator_loss*a	   �9��>   �9��>      �?!   �9��>) �a/\�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �W�U=2��|�~���MZ��K���������:              �?        �X���       b�D�	���~���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) Ʉ	'�{=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)  �"�I=2�u��gr��R%�������������:              �?        ���       b�D�	���~���A�*�
w
discriminator_loss*a	   �h�>   �h�>      �?!   �h�>)�a��k=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   �ء�   �ء�      �?!   �ء�)@8�.h�S=2��|�~���MZ��K���������:              �?        ��5��       b�D�	���~���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) �����=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   �ı�   �ı�      �?!   �ı�)@VQE�s=2��n�����豪}0ڰ��������:              �?        }�Y�       b�D�	Ħ�~���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ��� �=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �@���g=2���?�ګ�;9��R���������:              �?        P��I�       b�D�	5o�~���A�*�
w
discriminator_loss*a	   �j��>   �j��>      �?!   �j��>)���&��=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   �	P��   �	P��      �?!   �	P��) cSۣn=2豪}0ڰ�������������:              �?        ��?x�       b�D�	i��~���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  	�ww�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   @H��   @H��      �?!   @H��)�|��8B�=2�*��ڽ�G&�$���������:              �?        B*yS�       b�D�	��~���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>)@!�!r=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   @`��   @`��      �?!   @`��)�܀S(d=2���?�ګ�;9��R���������:              �?        ��%&�       b�D�	�������A�*�
w
discriminator_loss*a	   ��f�>   ��f�>      �?!   ��f�>) D`g���=2�f����>��(���>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��) $ ��W=2���]������|�~���������:              �?        ��l�       b�D�	������A�*�
w
discriminator_loss*a	   `ب>   `ب>      �?!   `ب>) -��Ic=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) $�L�XS=2��|�~���MZ��K���������:              �?        e2��       b�D�	2䳁���A�*�
w
discriminator_loss*a	   `\�>   `\�>      �?!   `\�>)@��4�l�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   @Ю�   @Ю�      �?!   @Ю�)� B�W�m=2豪}0ڰ�������������:              �?        ��3��       b�D�	�������A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �����w=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	    蠾    蠾      �?!    蠾)  t,�Q=2�MZ��K���u��gr���������:              �?        @����       b�D�	S�����A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)��L��0�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) Hd�$�=2G&�$��5�"�g����������:              �?        �>~�       b�D�	Á���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) � $b=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	    H��    H��      �?!    H��)  $O_=2�5�L�����]�����������:              �?        �/�       b�D�	�ǁ���A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>)�8!W�Vj=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) $�'0P=2�MZ��K���u��gr���������:              �?        l���       b�D�	��ˁ���A�*�
w
discriminator_loss*a	   �@�>   �@�>      �?!   �@�>)�ˉ4g=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) ��ub=2;9��R���5�L���������:              �?        d���       b�D�	�˵����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �P
q=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) 7@%OM=2�u��gr��R%�������������:              �?        �Fr��       b�D�	l������A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) d�o��Z=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@Z`�Q=2�MZ��K���u��gr���������:              �?        R:H��       b�D�	�Խ����A�*�
w
discriminator_loss*a	    $�>    $�>      �?!    $�>)  �1O��=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   �(��   �(��      �?!   �(��) y�Ek�[=2�5�L�����]�����������:              �?        �M��       b�D�	�����A�*�
w
discriminator_loss*a	   �P�>   �P�>      �?!   �P�>) $ ?��R=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)���"�A=239W$:���.��fc����������:              �?        ý��       b�D�	Ȅ���A�*�
w
discriminator_loss*a	   �d�>   �d�>      �?!   �d�>) �$V�%�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   �(��   �(��      �?!   �(��) x���c=2;9��R���5�L���������:              �?        l$c��       b�D�	�)̈́���A�*�
w
discriminator_loss*a	   �,��>   �,��>      �?!   �,��>) �mky��=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   �8��   �8��      �?!   �8��) ĀH�#\=2�5�L�����]�����������:              �?        ��%�       b�D�	Aф���A�*�
w
discriminator_loss*a	   `���>   `���>      �?!   `���>) �N��=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	    М�    М�      �?!    М�)   �K�I=2�u��gr��R%�������������:              �?        @`���       b�D�	'�Մ���A�*�
w
discriminator_loss*a	   �0�>   �0�>      �?!   �0�>) �t;�=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   �U�Ǿ   �U�Ǿ      �?!   �U�Ǿ)���I���=2
�/eq
Ⱦ����ž�������:              �?        .`2_�       b�D�	x㿇���A�*�
w
discriminator_loss*a	     �>     �>      �?!     �>) H ��c=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) y G�Z=2���]������|�~���������:              �?        �U��       b�D�	�(ć���A�*�
w
discriminator_loss*a	   `�>   `�>      �?!   `�>) a��#�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	    	t��    	t��      �?!    	t��) ���	s=2��n�����豪}0ڰ��������:              �?        ��p�       b�D�	�)ȇ���A�*�
w
discriminator_loss*a	   @ �>   @ �>      �?!   @ �>) � �H�[=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���) 	 �A�5=2
�}�����4[_>����������:              �?        �XA��       b�D�	<̇���A�*�
w
discriminator_loss*a	   @	|�>   @	|�>      �?!   @	|�>)����o�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	    	<��    	<��      �?!    	<��) �:R~Z�=25�"�g���0�6�/n���������:              �?        ԌOY�       b�D�	�WЇ���A�*�
w
discriminator_loss*a	   �
��>   �
��>      �?!   �
��>) r#�Tu�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) ɤ��Kr=2��n�����豪}0ڰ��������:              �?        J��       b�D�	C�ԇ���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) @�"�K=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	     ��     ��      �?!     ��) 
@�!0B=239W$:���.��fc����������:              �?        �ȃ�       b�D�	��؇���A�*�
w
discriminator_loss*a	   �д>   �д>      �?!   �д>)@(�H�{=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   `ॾ   `ॾ      �?!   `ॾ)@Z�~F�]=2�5�L�����]�����������:              �?        ,�)�       b�D�	��܇���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) r��J�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   �컾   �컾      �?!   �컾)�,�34]�=2�*��ڽ�G&�$���������:              �?        wo���       b�D�	X�܊���A�*�
w
discriminator_loss*a	   `h�>   `h�>      �?!   `h�>)@��B�z=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)�P�2k=2��������?�ګ��������:              �?        ��Ť�       b�D�	+^�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) Iv�h�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   `䰾   `䰾      �?!   `䰾)@��u��q=2��n�����豪}0ڰ��������:              �?        �4`�       b�D�	(4䊪��A�*�
w
discriminator_loss*a	   `X�>   `X�>      �?!   `X�>)@� ���Y=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   @p��   @p��      �?!   @p��)�(�ۋ�G=2R%������39W$:����������:              �?        Ϻ��       b�D�	�芪��A�*�
w
discriminator_loss*a	   �(�>   �(�>      �?!   �(�>) ��c��~=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   �`��   �`��      �?!   �`��) j�y=2�u`P+d����n������������:              �?        eb���       b�D�	��늪��A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�3C�U=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�(�$�I=2�u��gr��R%�������������:              �?        h|MG�       b�D�	:���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ����Os=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@Ρ�}V=2��|�~���MZ��K���������:              �?        ~��       b�D�	Y����A�*�
w
discriminator_loss*a	    @�>    @�>      �?!    @�>)  ���=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	    Ъ�    Ъ�      �?!    Ъ�) ,�CRwf=2���?�ګ�;9��R���������:              �?        \���       b�D�	�C�����A�*�
w
discriminator_loss*a	    @�>    @�>      �?!    @�>) b�И�l=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�T�&�K=2�u��gr��R%�������������:              �?        ��/t�       b�D�	UDэ���A�*�
w
discriminator_loss*a	   �د>   �د>      �?!   �د>)�l!o?�o=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  �[xd=2���?�ګ�;9��R���������:              �?        ���       b�D�	0 Ս���A�*�
w
discriminator_loss*a	   �h�>   �h�>      �?!   �h�>) ��_��w=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   `@��   `@��      �?!   `@��) ��[��`=2;9��R���5�L���������:              �?        XR���       b�D�	�ٍ���A�*�
w
discriminator_loss*a	    `�>    `�>      �?!    `�>) � DGvW=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�� V
O=2�MZ��K���u��gr���������:              �?        UX2�       b�D�	e�܍���A�*�
w
discriminator_loss*a	    (�>    (�>      �?!    (�>) Ȗ���=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) R�
 e=2���?�ګ�;9��R���������:              �?        �(��       b�D�	������A�*�
w
discriminator_loss*a	   �d�>   �d�>      �?!   �d�>)@�s�w=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) ���!R=2�MZ��K���u��gr���������:              �?        �x��       b�D�	2�䍪��A�*�
w
discriminator_loss*a	   �	\�>   �	\�>      �?!   �	\�>) ��V�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   �t��   �t��      �?!   �t��) ���	s=2��n�����豪}0ڰ��������:              �?        2�P~�       b�D�	E[荪��A�*�
w
discriminator_loss*a	   @	 �>   @	 �>      �?!   @	 �>)����.��=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   ` ��   ` ��      �?!   ` ��) 1:n=2豪}0ڰ�������������:              �?        �T��       b�D�	C8썪��A�*�
w
discriminator_loss*a	   ��~�>   ��~�>      �?!   ��~�>) ��6��=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)�lq�$c=2;9��R���5�L���������:              �?        P����       b�D�	ˁ䐪��A�*�
w
discriminator_loss*a	   @D�>   @D�>      �?!   @D�>) ��T��r=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	    
���    
���      �?!    
���) 4s��h=2���?�ګ�;9��R���������:              �?        1�,Z�       b�D�	 "ꐪ��A�*�
w
discriminator_loss*a	   @$�>   @$�>      �?!   @$�>) ��o��=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   �H��   �H��      �?!   �H��)@��&P�r=2��n�����豪}0ڰ��������:              �?        �y�z�       b�D�	����A�*�
w
discriminator_loss*a	    P�>    P�>      �?!    P�>)   �ˣN=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)����@=2.��fc���X$�z���������:              �?        ����       b�D�	�K�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@�ds}v=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�8�M=2�u��gr��R%�������������:              �?        p+;�       b�D�	\]�����A�*�
w
discriminator_loss*a	   �VP�>   �VP�>      �?!   �VP�>)@��L��=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)� bBa^k=2��������?�ګ��������:              �?        �5|��       b�D�	@������A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) �a�+j=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   �`��   �`��      �?!   �`��) DVJU=2��|�~���MZ��K���������:              �?        uNa�       b�D�	�����A�*�
w
discriminator_loss*a	    @�>    @�>      �?!    @�>) @8�K��=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) q)�ADz=20�6�/n���u`P+d���������:              �?        Ӎ��       b�D�	�C	����A�*�
w
discriminator_loss*a	   �	0�>   �	0�>      �?!   �	0�>) ��X�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) � �$S=2��|�~���MZ��K���������:              �?        �9��       b�D�	r�����A�*�
w
discriminator_loss*a	   �3f�>   �3f�>      �?!   �3f�>)@����=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   �ȣ�   �ȣ�      �?!   �ȣ�) y���tX=2���]������|�~���������:              �?        /�*��       b�D�	-�����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �A�*q�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) � ��i=2��������?�ګ��������:              �?        �K��       b�D�	������A�*�
w
discriminator_loss*a	   @t�>   @t�>      �?!   @t�>) A�ݦw=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   `l��   `l��      �?!   `l��)@f�#��p=2豪}0ڰ�������������:              �?        �j�N�       b�D�	'�"����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@��)�s=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   �H��   �H��      �?!   �H��) qH+�c=2;9��R���5�L���������:              �?        ,��>�       b�D�	'����A�*�
w
discriminator_loss*a	     �>     �>      �?!     �>) �  l=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   �ਾ   �ਾ      �?!   �ਾ) R*Vc=2;9��R���5�L���������:              �?        ��;�       b�D�	+����A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) Xܧ���=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	     ��     ��      �?!     ��)  �.=2���m!#���
�%W���������:              �?        {Ǖ�       b�D�	��/����A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>)�T@>l=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��)  � L=2�u��gr��R%�������������:              �?        ��˦�       b�D�	��5����A�*�
w
discriminator_loss*a	   �̹>   �̹>      �?!   �̹>) "���˄=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   `��   `��      �?!   `��)@�Kw~=20�6�/n���u`P+d���������:              �?        6@���       b�D�	��F����A�*�
w
discriminator_loss*a	   @
ܵ>   @
ܵ>      �?!   @
ܵ>) ��m�}=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) d��I�S=2��|�~���MZ��K���������:              �?        &�tI�       b�D�	ZfJ����A�*�
w
discriminator_loss*a	   �H�>   �H�>      �?!   �H�>)@�eƿ6�=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��)@H@��[=2�5�L�����]�����������:              �?        <���       b�D�	4N����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) ��1�>2O�ʗ��>>�?�s��>�������:              �?        
s
generator_loss*a	     ��     ��      �?!     ��) bE�8\l=2��������?�ګ��������:              �?        �
}��       �N�	�<Q����A*�
w
discriminator_loss*a	    6�>    6�>      �?!    6�>)@�v٥��=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   �,��   �,��      �?!   �,��)@�yƳ�t=2��n�����豪}0ڰ��������:              �?        �Z�z�       �{�	�R����A*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) i!O��v=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)@� ���W=2���]������|�~���������:              �?        �����       �{�	��V����A
*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@���}=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) I�T�s=2��n�����豪}0ڰ��������:              �?        �9i�       �{�	��Z����A*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) D�OMx=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��)�l��,\l=2��������?�ګ��������:              �?        �y�9�       �{�	B�^����A*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) 9̰D\�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) y��XZ=2���]������|�~���������:              �?        ����       �{�	ߪc����A*�
w
discriminator_loss*a	    \�>    \�>      �?!    \�>) ��"�p=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	    X��    X��      �?!    X��) Ȁ�z]g=2���?�ګ�;9��R���������:              �?        �m��       �{�	�h����A*�
w
discriminator_loss*a	   `,�>   `,�>      �?!   `,�>)@�G�v=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   @Н�   @Н�      �?!   @Н�)�T@N�K=2�u��gr��R%�������������:              �?        NO���       �{�	�m����A#*�
w
discriminator_loss*a	   @ �>   @ �>      �?!   @ �>)�T��$0b=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) � ]=2�5�L�����]�����������:              �?        �ꁢ�       �{�	�q����A(*�
w
discriminator_loss*a	    Э>    Э>      �?!    Э>) � QQ�k=2���?�ګ>����>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)   �#J=2�u��gr��R%�������������:              �?        ;$��       �{�	k������A-*�
w
discriminator_loss*a	   � �>   � �>      �?!   � �>)@��@t=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�DO��m=2豪}0ڰ�������������:              �?        �P��       �{�	�v�����A2*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) 7��>L=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	   @Г�   @Г�      �?!   @Г�) ���8=2X$�z��
�}�����������:              �?        �
'�       �{�	Xy�����A7*�
w
discriminator_loss*a	   `�>   `�>      �?!   `�>)@�iOi��=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	    p��    p��      �?!    p��) ��s=2��n�����豪}0ڰ��������:              �?        ����       �{�	������A<*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �\�5�=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@��ǞNq=2豪}0ڰ�������������:              �?        �
bK�       �{�	X䠝���AA*�
w
discriminator_loss*a	   �P�>   �P�>      �?!   �P�>) d��=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���) [����O=2�MZ��K���u��gr���������:              �?        �����       �{�	������AF*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) "dݤ�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  L��_=2�5�L�����]�����������:              �?        �^A��       �{�	Mɨ����AK*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) � i�q=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   `p��   `p��      �?!   `p��)@@�S=2��|�~���MZ��K���������:              �?        2B3�       �{�	欝���AP*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) ��w)�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   �྾   �྾      �?!   �྾) VS Pʍ=2�[�=�k���*��ڽ��������:              �?        P��P�       �{�	�頪��AU*�
w
discriminator_loss*a	   @ �>   @ �>      �?!   @ �>)�8�
�f=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   @ ��   @ ��      �?!   @ ��)�� �HJ=2�u��gr��R%�������������:              �?        U���       �{�	Y���AZ*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) B!�Nn�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) B11�>�=2�[�=�k���*��ڽ��������:              �?        ��Nl�       �{�	�����A_*�
w
discriminator_loss*a	   �	t�>   �	t�>      �?!   �	t�>) �Ź�	s=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   �ؠ�   �ؠ�      �?!   �ؠ�) 1@�g�Q=2�MZ��K���u��gr���������:              �?        ����       �{�	������Ad*�
w
discriminator_loss*a	    4��>    4��>      �?!    4��>)  ����=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)@8�=(vT=2��|�~���MZ��K���������:              �?        ����       �{�	׸�����Ai*�
w
discriminator_loss*a	    �>    �>      �?!    �>) Ȁ�Vj=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) �PI�b=2;9��R���5�L���������:              �?        F����       �{�	������An*�
w
discriminator_loss*a	   �"�>   �"�>      �?!   �"�>)����/�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   @P��   @P��      �?!   @P��)�(@�ˢE=2R%������39W$:����������:              �?        �&���       �{�	Ê����As*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) � �\=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���) 	 �AD6=2
�}�����4[_>����������:              �?        �9��       �{�	�]����Ax*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) �d̕�v=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) �FO�Vj=2��������?�ګ��������:              �?        Q��=�       �{�	�/:����A}*�
w
discriminator_loss*a	   @l��>   @l��>      �?!   @l��>) a�^��=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) Q��I�Z=2���]������|�~���������:              �?        ���|�       b�D�	��>����A�*�
w
discriminator_loss*a	   `fA�>   `fA�>      �?!   `fA�>)@
[�1Դ=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   ��   ��      �?!   ��) �oc=2;9��R���5�L���������:              �?         3���       b�D�	N)C����A�*�
w
discriminator_loss*a	    h�>    h�>      �?!    h�>)  ��+d=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   �X��   �X��      �?!   �X��)@��c��Y=2���]������|�~���������:              �?        ��~��       b�D�	�G����A�*�
w
discriminator_loss*a	   � ��>   � ��>      �?!   � ��>)� }��/=2�
�%W�>���m!#�>�������:              �?        
s
generator_loss*a	   @ ���   @ ���      �?!   @ ���)  � �=2K���7��[#=�؏���������:              �?        ���       b�D�	o�K����A�*�
w
discriminator_loss*a	   �O��>   �O��>      �?!   �O��>)@B�]��=2K+�E���>jqs&\��>�������:              �?        
s
generator_loss*a	   `��̾   `��̾      �?!   `��̾) ���K!�=2['�?�;;�"�qʾ�������:              �?        ~���       b�D�	�O����A�*�
w
discriminator_loss*a	   `p�>   `p�>      �?!   `p�>)@Z@-S=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���) 	 �AF1=2�4[_>������m!#���������:              �?        Ss���       b�D�		9T����A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�L����=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�@(a!m=2��������?�ګ��������:              �?        ���       b�D�	sX����A�*�
w
discriminator_loss*a	   @x�>   @x�>      �?!   @x�>) ��QU=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	    p��    p��      �?!    p��)   .�8D=239W$:���.��fc����������:              �?        �����       b�D�	ͬ�����A�*�
w
discriminator_loss*a	   �0�>   �0�>      �?!   �0�>)@n�G��^=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) b �&\L=2�u��gr��R%�������������:              �?        ��WA�       b�D�	o������A�*�
w
discriminator_loss*a	   `4�>   `4�>      �?!   `4�>)@2qܱhp=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   ��   ��      �?!   ��)�p@|�O=2�MZ��K���u��gr���������:              �?        �3{a�       b�D�	�_�����A�*�
w
discriminator_loss*a	   `X�>   `X�>      �?!   `X�>) [��w]g=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	    ئ�    ئ�      �?!    ئ�) H�H�N`=2�5�L�����]�����������:              �?        �Mo��       b�D�	7(�����A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>)�� ��Vj=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   �(��   �(��      �?!   �(��) ��bn�^=2�5�L�����]�����������:              �?        �2��       b�D�	j󡧪��A�*�
w
discriminator_loss*a	   �8�>   �8�>      �?!   �8�>)������`=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	    8��    8��      �?!    8��)  �̇R=2�MZ��K���u��gr���������:              �?        �E��       b�D�	�̥����A�*�
w
discriminator_loss*a	    h�>    h�>      �?!    h�>) ȃ�e7i=2���?�ګ>����>�������:              �?        
s
generator_loss*a	    h��    h��      �?!    h��) �S\d�e=2���?�ګ�;9��R���������:              �?        ��U�       b�D�	�������A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>)@A`l^=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   @ ��   @ ��      �?!   @ ��)�� �	 L=2�u��gr��R%�������������:              �?        �0gs�       b�D�	3a�����A�*�
w
discriminator_loss*a	   @e,�>   @e,�>      �?!   @e,�>)�\�bh�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	    exǾ    exǾ      �?!    exǾ) Ⱦ&�6�=2
�/eq
Ⱦ����ž�������:              �?        ֻn��       b�D�	�֪���A�*�
w
discriminator_loss*a	   �
�>   �
�>      �?!   �
�>) �}�#�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) 1���;=2X$�z��
�}�����������:              �?        ���a�       b�D�	{V۪���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) k��r��=2�ѩ�-�>���%�>�������:              �?        
s
generator_loss*a	    Ȣ�    Ȣ�      �?!    Ȣ�)  d�V=2��|�~���MZ��K���������:              �?        �����       b�D�	�᪪��A�*�
w
discriminator_loss*a	    8�>    8�>      �?!    8�>) �����=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)��GA��g=2���?�ګ�;9��R���������:              �?        |p��       b�D�	��䪪��A�*�
w
discriminator_loss*a	   � �>   � �>      �?!   � �>)@�L�{=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	    @��    @��      �?!    @��)  h)7=2
�}�����4[_>����������:              �?        ^f���       b�D�	��說��A�*�
w
discriminator_loss*a	   �ȯ>   �ȯ>      �?!   �ȯ>) ��v��o=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   `К�   `К�      �?!   `К�) [�MwF=2R%������39W$:����������:              �?        �e���       b�D�	�L�����A�*�
w
discriminator_loss*a	   @а>   @а>      �?!   @а>) q�"��q=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   ��   ��      �?!   ��) d �T=2��|�~���MZ��K���������:              �?        F-��       b�D�	Ac���A�*�
w
discriminator_loss*a	   �&X�>   �&X�>      �?!   �&X�>) R�+��=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��) A��wH�=25�"�g���0�6�/n���������:              �?        �;��       b�D�	������A�*�
w
discriminator_loss*a	   �6X�>   �6X�>      �?!   �6X�>)@4\D}4�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   �  ��   �  ��      �?!   �  ��)@�jA�4=2
�}�����4[_>����������:              �?        �g<<�       b�D�	!P����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) y G�Z=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�T�&�K=2�u��gr��R%�������������:              �?        ����       b�D�	XhT����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �h:}=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���) �����O=2�MZ��K���u��gr���������:              �?        *�~��       b�D�	��X����A�*�
w
discriminator_loss*a	   ��r�>   ��r�>      �?!   ��r�>)@zڎ�!�=2��~]�[�>��>M|K�>�������:              �?        
s
generator_loss*a	   �П�   �П�      �?!   �П�)�<�wM�O=2�MZ��K���u��gr���������:              �?        V����       b�D�	��\����A�*�
w
discriminator_loss*a	   @L�>   @L�>      �?!   @L�>) I�y�t=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@*�3��:=2X$�z��
�}�����������:              �?        y]���       b�D�	�C`����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) w�����=2['�?��>K+�E���>�������:              �?        
s
generator_loss*a	   �	 ��   �	 ��      �?!   �	 ��) ��Z�^=2�5�L�����]�����������:              �?        �B���       b�D�	�d����A�*�
w
discriminator_loss*a	    h�>    h�>      �?!    h�>)@��к�p=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   �`��   �`��      �?!   �`��)  I"�B=239W$:���.��fc����������:              �?        P(��       b�D�	��g����A�*�
w
discriminator_loss*a	    Ȣ>    Ȣ>      �?!    Ȣ>)@H ��V=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��) H C�H=2R%������39W$:����������:              �?        悾��       b�D�	k�k����A�*�
w
discriminator_loss*a	    �p�>    �p�>      �?!    �p�>)@XU�;y�=2I��P=�>��Zr[v�>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���)@�[==2.��fc���X$�z���������:              �?         =��       b�D�	
߶����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@� JxQ=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	   �P��   �P��      �?!   �P��) ���c<=2X$�z��
�}�����������:              �?        �~�       b�D�	麺����A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@��hGs=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��) 1���`P=2�MZ��K���u��gr���������:              �?        ��U8�       b�D�	������A�*�
w
discriminator_loss*a	   @0�>   @0�>      �?!   @0�>)����OHb=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	     ��     ��      �?!     ��)  p� =2��z!�?��T�L<���������:              �?        �a���       b�D�	�}±���A�*�
w
discriminator_loss*a	   �#��>   �#��>      �?!   �#��>) ���^�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	    0��    0��      �?!    0��)@,���xY=2���]������|�~���������:              �?        ����       b�D�	�eƱ���A�*�
w
discriminator_loss*a	   �$�>   �$�>      �?!   �$�>) ��dZy=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   �
���   �
���      �?!   �
���) �SJR�b=2;9��R���5�L���������:              �?        ��q��       b�D�	�.ʱ���A�*�
w
discriminator_loss*a	   �|�>   �|�>      �?!   �|�>) � ���p=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   `覾   `覾      �?!   `覾) ��e`=2�5�L�����]�����������:              �?        �Q[�       b�D�	F�ͱ���A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) ���+j=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   `p��   `p��      �?!   `p��) E����b=2;9��R���5�L���������:              �?        ����       b�D�	}ұ���A�*�
w
discriminator_loss*a	     �>     �>      �?!     �>)   @t=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��)��O7�f=2���?�ګ�;9��R���������:              �?        ,�:G�       b�D�	7%����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�p@m�`O=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	    Ж�    Ж�      �?!    Ж�) H GLC@=2.��fc���X$�z���������:              �?        ��N�       b�D�	�)����A�*�
w
discriminator_loss*a	   @0�>   @0�>      �?!   @0�>)� Une=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���) 	 ��5=2
�}�����4[_>����������:              �?        0�]Y�       b�D�	��,����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) � ��Z=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��) y�@��T=2��|�~���MZ��K���������:              �?        �&X0�       b�D�	�31����A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>)  q	�@=2X$�z�>.��fc��>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���) 	���g1=2�4[_>������m!#���������:              �?        �d=�       b�D�	|�5����A�*�
w
discriminator_loss*a	   � �>   � �>      �?!   � �>) r�!��=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  ���m=2豪}0ڰ�������������:              �?        �ZE�       b�D�	Ң9����A�*�
w
discriminator_loss*a	   @9F�>   @9F�>      �?!   @9F�>)�l���i�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   �%4��   �%4��      �?!   �%4��) ��~=20�6�/n���u`P+d���������:              �?        �I�Q�       b�D�	�f=����A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) �:�[{=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�(@��kD=239W$:���.��fc����������:              �?        K����       b�D�	�A����A�*�
w
discriminator_loss*a	     ��>     ��>      �?!     ��>)@����/�=2�XQ��>�����>�������:              �?        
s
generator_loss*a	   �p��   �p��      �?!   �p��)�dB�8d=2���?�ګ�;9��R���������:              �?        ?6��       b�D�	�ꚸ���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  �
�\=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   �`��   �`��      �?!   �`��) B@�$�E=2R%������39W$:����������:              �?        ���       b�D�	@=�����A�*�
w
discriminator_loss*a	    eX�>    eX�>      �?!    eX�>) �o��=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	     ��     ��      �?!     ��)  �BP9=2X$�z��
�}�����������:              �?        e�A��       b�D�	�פ����A�*�
w
discriminator_loss*a	    x�>    x�>      �?!    x�>)  ML�g=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) N`��mM=2�u��gr��R%�������������:              �?        ܃^�       b�D�	`������A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) <��q�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�(����G=2R%������39W$:����������:              �?        ��6��       b�D�	�������A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) ���
y=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   �4��   �4��      �?!   �4��)@���hp=2豪}0ڰ�������������:              �?        ��3�       b�D�	kZ�����A�*�
w
discriminator_loss*a	   �  �>   �  �>      �?!   �  �>)  "A�4=2�4[_>��>
�}���>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���)� 5��$=2��ӤP�����z!�?���������:              �?        c�̞�       b�D�	�*�����A�*�
w
discriminator_loss*a	   �f��>   �f��>      �?!   �f��>) R��Z�=2�����>
�/eq
�>�������:              �?        
s
generator_loss*a	   @@��   @@��      �?!   @@��)�( ���C=239W$:���.��fc����������:              �?        [};�       b�D�	�������A�*�
w
discriminator_loss*a	    X�>    X�>      �?!    X�>) � �wa=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) Ā���W=2���]������|�~���������:              �?        DN+:�       b�D�	&U����A�*�
w
discriminator_loss*a	   `P�>   `P�>      �?!   `P�>) -��OG=239W$:��>R%�����>�������:              �?        
s
generator_loss*a	    0��    0��      �?!    0��) @ ���>=2.��fc���X$�z���������:              �?        _����       b�D�	�����A�*�
w
discriminator_loss*a	   �P�>   �P�>      �?!   �P�>)��tѢe=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   �  ��   �  ��      �?!   �  ��) 	 �A@0=2���m!#���
�%W���������:              �?        [�</�       b�D�	������A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ���vU=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   @�   @�      �?!   @�)�D�|%�M=2�u��gr��R%�������������:              �?        �gL��       b�D�	[�����A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)�؉\C�o=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   �@��   �@��      �?!   �@��)��	s��j=2��������?�ګ��������:              �?        o�g �       b�D�	l����A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) ��#ej=2���?�ګ>����>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��)  @ B=239W$:���.��fc����������:              �?        ��E��       b�D�	������A�*�
w
discriminator_loss*a	   �d�>   �d�>      �?!   �d�>) �b4��y=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) ��@��d=2���?�ګ�;9��R���������:              �?        Z�4��       b�D�	�� ����A�*�
w
discriminator_loss*a	   �l�>   �l�>      �?!   �l�>)@r�8u�w=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   @L��   @L��      �?!   @L��) ����p=2豪}0ڰ�������������:              �?        [��$�       b�D�	%����A�*�
w
discriminator_loss*a	    4$�>    4$�>      �?!    4$�>) �%w6�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  �B�=2T�L<��u��6
���������:              �?        ��X��       b�D�	��y����A�*�
w
discriminator_loss*a	   @H�>   @H�>      �?!   @H�>)���,0�n=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	    X��    X��      �?!    X��) � !�P=2�MZ��K���u��gr���������:              �?        �!��       b�D�	7�~����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@
�W=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@��C�==2.��fc���X$�z���������:              �?        R塙�       b�D�	�h�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ����-�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   @謾   @謾      �?!   @謾)� ��j=2��������?�ګ��������:              �?        !���       b�D�	�⇿���A�*�
w
discriminator_loss*a	    @�>    @�>      �?!    @�>)  	`�y=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) r���k=2��������?�ګ��������:              �?        >']�       b�D�	p�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@8�[��U=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   �И�   �И�      �?!   �И�)���J=C=239W$:���.��fc����������:              �?        ����       b�D�	�v�����A�*�
w
discriminator_loss*a	   �س>   �س>      �?!   �س>) �Q���x=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@FqW).t=2��n�����豪}0ڰ��������:              �?        K����       b�D�	:������A�*�
w
discriminator_loss*a	   @P�>   @P�>      �?!   @P�>)��Ax�d=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)���(+�b=2;9��R���5�L���������:              �?        �����       b�D�	�U�����A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>) -P_%!`=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) $`i˞D=239W$:���.��fc����������:              �?        L<��       b�D�	��ª��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��!]=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   `У�   `У�      �?!   `У�)@��ɟ�X=2���]������|�~���������:              �?        T�5a�       b�D�	��ª��A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@H�~��:=2
�}���>X$�z�>�������:              �?        
s
generator_loss*a	    P��    P��      �?!    P��) @ ԔO7=2
�}�����4[_>����������:              �?        �W��       b�D�	*�ª��A�*�
w
discriminator_loss*a	   �~�>   �~�>      �?!   �~�>) D(����=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) rn�Fi�=2�*��ڽ�G&�$���������:              �?        ����       b�D�	z��ª��A�*�
w
discriminator_loss*a	   �ȫ>   �ȫ>      �?!   �ȫ>) S��ph=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   �P��   �P��      �?!   �P��) B`D��J=2�u��gr��R%�������������:              �?        
	�       b�D�	�d�ª��A�*�
w
discriminator_loss*a	   `0/�>   `0/�>      �?!   `0/�>) !c>���=2�iD*L��>E��a�W�>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���)@Z 2$S=2��|�~���MZ��K���������:              �?        Y��N�       b�D�	��ª��A�*�
w
discriminator_loss*a	   @�P�>   @�P�>      �?!   @�P�>) y>5�e�=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	   �  ��   �  ��      �?!   �  ��)  � T%=2��ӤP�����z!�?���������:              �?        #6L�       b�D�	A��ª��A�*�
w
discriminator_loss*a	   � �>   � �>      �?!   � �>)@V@	 p=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   �ࣾ   �ࣾ      �?!   �ࣾ)@V�}K�X=2���]������|�~���������:              �?        v_oG�       b�D�	ŕê��A�*�
w
discriminator_loss*a	   @&L�>   @&L�>      �?!   @&L�>) q�붙�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)����'�F=2R%������39W$:����������:              �?        f��t�       b�D�	Z�Wƪ��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) d hDZ=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) d JFPY=2���]������|�~���������:              �?        e`��       b�D�	ȣ[ƪ��A�*�
w
discriminator_loss*a	    O@�>    O@�>      �?!    O@�>) ü�`�=2
�/eq
�>;�"�q�>�������:              �?        
s
generator_loss*a	   �@��   �@��      �?!   �@��)@* ��2=2�4[_>������m!#���������:              �?        ��+�       b�D�	Dt_ƪ��A�*�
w
discriminator_loss*a	   �t�>   �t�>      �?!   �t�>) �k�Dޅ=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   @x��   @x��      �?!   @x��)�X�XEd=2���?�ګ�;9��R���������:              �?        !�)F�       b�D�	�?cƪ��A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>) ���+�V=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��) 1�ڔ�>=2.��fc���X$�z���������:              �?        ��=��       b�D�	�gƪ��A�*�
w
discriminator_loss*a	    Ƞ>    Ƞ>      �?!    Ƞ>) �ҙQ=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)���!Z =2��z!�?��T�L<���������:              �?        ���       b�D�	��jƪ��A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@� �aX=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	    ࡾ    ࡾ      �?!    ࡾ) � �F�S=2��|�~���MZ��K���������:              �?        L���       b�D�	<oƪ��A�*�
w
discriminator_loss*a	    ԰>    ԰>      �?!    ԰>)@,���q=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) H ��OK=2�u��gr��R%�������������:              �?        ���l�       b�D�	�sƪ��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) � �(AO=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)����&�B=239W$:���.��fc����������:              �?        )4`�       b�D�	���ɪ��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) � ��$c=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��) � f2R=2�MZ��K���u��gr���������:              �?        ����       b�D�	d��ɪ��A�*�
w
discriminator_loss*a	   � �>   � �>      �?!   � �>) 	��(9=2
�}���>X$�z�>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���)��B!�&=2�
�%W����ӤP����������:              �?        ��2��       b�D�	���ɪ��A�*�
w
discriminator_loss*a	    !i�>    !i�>      �?!    !i�>) d!a��=2�uE����>�f����>�������:              �?        
s
generator_loss*a	   `0��   `0��      �?!   `0��) c' bg=2���?�ګ�;9��R���������:              �?        %f'I�       b�D�	���ɪ��A�*�
w
discriminator_loss*a	   �QB�>   �QB�>      �?!   �QB�>) ��Jִ=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	   `Д�   `Д�      �?!   `Д�)@���;=2X$�z��
�}�����������:              �?        ����       b�D�	J��ɪ��A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@`׆Q=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)�<@�̷I=2�u��gr��R%�������������:              �?        i���       b�D�	݀�ɪ��A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ��<��w=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   ��   ��      �?!   ��) R���F=2R%������39W$:����������:              �?        ����       b�D�	MU�ɪ��A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �v���=2��>M|K�>�_�T�l�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) ��epi=2��������?�ګ��������:              �?        �6�       b�D�	/��ɪ��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) $ �CD6=2�4[_>��>
�}���>�������:              �?        
s
generator_loss*a	   �@��   �@��      �?!   �@��) $ �0=2�4[_>������m!#���������:              �?        �"��       b�D�	��>ͪ��A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)  aC:==2X$�z�>.��fc��>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)@ �3=2�4[_>������m!#���������:              �?        ���       b�D�	��Bͪ��A�*�
w
discriminator_loss*a	    Й>    Й>      �?!    Й>) $�mK�D=2.��fc��>39W$:��>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��)   "�C=239W$:���.��fc����������:              �?        r�0��       b�D�	oGͪ��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@r�r�p{=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��) $ ��7=2
�}�����4[_>����������:              �?        {���       b�D�	n�Kͪ��A�*�
w
discriminator_loss*a	   �$l�>   �$l�>      �?!   �$l�>) �aE��=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���) $ �(9=2X$�z��
�}�����������:              �?        Ϳ�^�       b�D�	��Oͪ��A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) �כ\t=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	    `��    `��      �?!    `��)  �!�,=2���m!#���
�%W���������:              �?        �W�       b�D�	|�Sͪ��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@8@�gQ=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) 1 D�5=2
�}�����4[_>����������:              �?        �芈�       b�D�	��Wͪ��A�*�
w
discriminator_loss*a	   �X�>   �X�>      �?!   �X�>)@\����r=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	    @��    @��      �?!    @��)  h)7=2
�}�����4[_>����������:              �?        �e���       b�D�	�V]ͪ��A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�8��+�K=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ]	RD=239W$:���.��fc����������:              �?        yPҟ�       b�D�	��Ъ��A�*�
w
discriminator_loss*a	   `�>   `�>      �?!   `�>)@`�P=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	   ��   ��      �?!   ��) 2��+J=2�u��gr��R%�������������:              �?        ^��       b�D�	�Ъ��A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>) Q���8=2
�}���>X$�z�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) x��&�(=2�
�%W����ӤP����������:              �?        ͈	�       b�D�	���Ъ��A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) �N�n=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   �Ȭ�   �Ȭ�      �?!   �Ȭ�)����i=2��������?�ګ��������:              �?        ���       b�D�	��Ъ��A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) y�5dT=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   �`��   �`��      �?!   �`��)@n��E�R=2��|�~���MZ��K���������:              �?        ���#�       b�D�	�<�Ъ��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�� �Rd=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   @ॾ   @ॾ      �?!   @ॾ) !�K�]=2�5�L�����]�����������:              �?        C�I/�       b�D�	���Ъ��A�*�
w
discriminator_loss*a	   �Ȩ>   �Ȩ>      �?!   �Ȩ>) �@��0c=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	   ��   ��      �?!   ��) �JL=2�u��gr��R%�������������:              �?        �'�4�       b�D�	���Ъ��A�*�
w
discriminator_loss*a	   � p�>   � p�>      �?!   � p�>) 	��3=2���m!#�>�4[_>��>�������:              �?        
s
generator_loss*a	   `���   `���      �?!   `���) @�"�/=2���m!#���
�%W���������:              �?        ����       b�D�	���Ъ��A�*�
w
discriminator_loss*a	   @Ȣ>   @Ȣ>      �?!   @Ȣ>) Q@H�V=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   � `��   � `��      �?!   � `��)  � �%=2��ӤP�����z!�?���������:              �?        ��5��       b�D�	AX�Ԫ��A�*�
w
discriminator_loss*a	   �~��>   �~��>      �?!   �~��>)@n�[��=2jqs&\��>��~]�[�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) � ��@=2.��fc���X$�z���������:              �?        8� �       b�D�	iw�Ԫ��A�*�
w
discriminator_loss*a	   �x�>   �x�>      �?!   �x�>)@*`��S=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	    И�    И�      �?!    И�)   K=C=239W$:���.��fc����������:              �?        ���X�       b�D�	�I�Ԫ��A�*�
w
discriminator_loss*a	   �М>   �М>      �?!   �М>)�<��L�I=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��)  Bd4=2
�}�����4[_>����������:              �?        l`]�       b�D�	�"�Ԫ��A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) ����`=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	   � �   � �      �?!   � �) 	��4=2
�}�����4[_>����������:              �?        7�;��       b�D�	��Ԫ��A�*�
w
discriminator_loss*a	   �(<�>   �(<�>      �?!   �(<�>)@&�Q�~=2�u`P+d�>0�6�/n�>�������:              �?        
s
generator_loss*a	    `��    `��      �?!    `��) 
��!))=2�
�%W����ӤP����������:              �?         ;dM�       b�D�	ÓԪ��A�*�
w
discriminator_loss*a	    0�>    0�>      �?!    0�>) @ ���T=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) 1 �D�;=2X$�z��
�}�����������:              �?        �P�       b�D�	K��Ԫ��A�*�
w
discriminator_loss*a	   �`�>   �`�>      �?!   �`�>) x@�&)I=239W$:��>R%�����>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) i ��C=239W$:���.��fc����������:              �?        tY��       b�D�	E��Ԫ��A�*�
w
discriminator_loss*a	   �`�>   �`�>      �?!   �`�>) � }JJ_=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   �@��   �@��      �?!   �@��) � l)W=2���]������|�~���������:              �?        b>#B�       b�D�		��ת��A�*�
w
discriminator_loss*a	   `H�>   `H�>      �?!   `H�>) 1�_2�n=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   `(��   `(��      �?!   `(��)@��qdY=2���]������|�~���������:              �?        �=l��       b�D�	��ת��A�*�
w
discriminator_loss*a	   �#��>   �#��>      �?!   �#��>)@p0��5�=2�[�=�k�>��~���>�������:              �?        
s
generator_loss*a	   �F��   �F��      �?!   �F��) �;��>�=2�[�=�k���*��ڽ��������:              �?        >%�*�       b�D�	���ת��A�*�
w
discriminator_loss*a	    &\�>    &\�>      �?!    &\�>)  �����=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��)@@f22=2�4[_>������m!#���������:              �?        .�8�       b�D�	G��ת��A�*�
w
discriminator_loss*a	   �@�>   �@�>      �?!   �@�>) b ֆ�N=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	   ``��   ``��      �?!   ``��) [@�%)I=2R%������39W$:����������:              �?        �O���       b�D�	\ت��A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) HH�Q�=2�_�T�l�>�iD*L��>�������:              �?        
s
generator_loss*a	   `p��   `p��      �?!   `p��) ��R��L=2�u��gr��R%�������������:              �?        ���       b�D�	�!ت��A�*�
w
discriminator_loss*a	   `(�>   `(�>      �?!   `(�>) ��t��c=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@rh3_�U=2��|�~���MZ��K���������:              �?        (��f�       b�D�	�@ت��A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�T�A%�D=2.��fc��>39W$:��>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) H J$Z@=2.��fc���X$�z���������:              �?        D��A�       b�D�	a8ت��A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) R��Ge=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   @0��   @0��      �?!   @0��) �ג�4=2
�}�����4[_>����������:              �?        M����       b�D�	^�~۪��A�*�
w
discriminator_loss*a	   �H�>   �H�>      �?!   �H�>)@��cY<w=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��)��� l=2��������?�ګ��������:              �?        c�2T�       b�D�	
��۪��A�*�
w
discriminator_loss*a	   `	��>   `	��>      �?!   `	��>)@~E���s=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  xa8=2X$�z��
�}�����������:              �?        ��í�       b�D�	I��۪��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@���f�_=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   � @��   � @��      �?!   � @��)� �� =2��z!�?��T�L<���������:              �?        �����       b�D�	�t�۪��A�*�
w
discriminator_loss*a	   �H�>   �H�>      �?!   �H�>)@��&P�r=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   @0��   @0��      �?!   @0��) Q�e�W=2���]������|�~���������:              �?        �J��       b�D�	>�۪��A�*�
w
discriminator_loss*a	   @ �>   @ �>      �?!   @ �>) �5MPY=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)  aC:==2.��fc���X$�z���������:              �?        �ƪ�       b�D�	��۪��A�*�
w
discriminator_loss*a	   �`�>   �`�>      �?!   �`�>) q)���y=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  TB�5=2
�}�����4[_>����������:              �?        �]��       b�D�	%�۪��A�*�
w
discriminator_loss*a	   �@�>   �@�>      �?!   �@�>) 2 j��H=239W$:��>R%�����>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) 1 �D<;=2X$�z��
�}�����������:              �?        ��8�       b�D�	A�۪��A�*�
w
discriminator_loss*a	   �Ȧ>   �Ȧ>      �?!   �Ȧ>) �Q��7`=2���]���>�5�L�>�������:              �?        
s
generator_loss*a	   @Ж�   @Ж�      �?!   @Ж�)��@NC@=2.��fc���X$�z���������:              �?        w�C,�       b�D�	8\ߪ��A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�T@�OK=2R%�����>�u��gr�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) ���5=2
�}�����4[_>����������:              �?        ~����       b�D�	D1ߪ��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) i���VQ=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  H��B=239W$:���.��fc����������:              �?        lCW��       b�D�	rߪ��A�*�
w
discriminator_loss*a	   �ܸ>   �ܸ>      �?!   �ܸ>) ��kP�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)@*@��5=2
�}�����4[_>����������:              �?        Hlc�       b�D�	��ߪ��A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@h	��=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	    `��    `��      �?!    `��) H �$�E=2R%������39W$:����������:              �?        d�F�       b�D�	~�ߪ��A�*�
w
discriminator_loss*a	   @�>   @�>      �?!   @�>) ���Qvt=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   @ ��   @ ��      �?!   @ ��) �+r=2��n�����豪}0ڰ��������:              �?        ���       b�D�	+U"ߪ��A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) � w2R=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) � �D:=2X$�z��
�}�����������:              �?        {B�       b�D�	�#&ߪ��A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)  FdH=239W$:��>R%�����>�������:              �?        
s
generator_loss*a	   ���   ���      �?!   ���)���
�@=2.��fc���X$�z���������:              �?        @m��       b�D�	k#*ߪ��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) d �EjS=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���)  X�5=2
�}�����4[_>����������:              �?        �3m�       b�D�	�Q����A�*�
w
discriminator_loss*a	    `�>    `�>      �?!    `�>)  `SvW=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���)� 5��$=2��ӤP�����z!�?���������:              �?        �\��       b�D�	�����A�*�
w
discriminator_loss*a	   �#H�>   �#H�>      �?!   �#H�>) ����=2��~���>�XQ��>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) $ �CP9=2X$�z��
�}�����������:              �?        D~�*�       b�D�	�����A�*�
w
discriminator_loss*a	    Ю>    Ю>      �?!    Ю>) �Kt�m=2����>豪}0ڰ>�������:              �?        
s
generator_loss*a	   ` ��   ` ��      �?!   ` ��)@Z ��6=2
�}�����4[_>����������:              �?        $><�       �N�	������A*�
w
discriminator_loss*a	     �>     �>      �?!     �>) ,�	 E=239W$:��>R%�����>�������:              �?        
s
generator_loss*a	   �@��   �@��      �?!   �@��) � ��(=2�
�%W����ӤP����������:              �?        P�F+�       �{�	g�%���A*�
w
discriminator_loss*a	   �4�>   �4�>      �?!   �4�>) 2�Nm�=2�*��ڽ>�[�=�k�>�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���)�T�o�`O=2�MZ��K���u��gr���������:              �?        t/��       �{�	�*���A
*�
w
discriminator_loss*a	    �>    �>      �?!    �>)@H�EDV=2�MZ��K�>��|�~�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��)�< L�C=239W$:���.��fc����������:              �?        �T���       �{�	�.���A*�
w
discriminator_loss*a	   �$�>   �$�>      �?!   �$�>) ���\r=2豪}0ڰ>��n����>�������:              �?        
s
generator_loss*a	   �ث�   �ث�      �?!   �ث�) �E~I:h=2���?�ګ�;9��R���������:              �?        ֺiq�       �{�	��2���A*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �����Z=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��) b��MG=2R%������39W$:����������:              �?        x'��       �{�	�_6���A*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)��跉
�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	    0��    0��      �?!    0��)@,��W=2���]������|�~���������:              �?        Ș1X�       �{�	G0:���A*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) ��|�`=2�5�L�>;9��R�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) y �E@P=2�MZ��K���u��gr���������:              �?        D���       �{�	Q�=���A#*�
w
discriminator_loss*a	    P�>    P�>      �?!    P�>) H �̢E=239W$:��>R%�����>�������:              �?        
s
generator_loss*a	   �0��   �0��      �?!   �0��)@n���x9=2X$�z��
�}�����������:              �?        �G/�       �{�	�B���A(*�
w
discriminator_loss*a	    &ػ>    &ػ>      �?!    &ػ>) l�Xt:�=2G&�$�>�*��ڽ>�������:              �?        
s
generator_loss*a	   `@��   `@��      �?!   `@��) [����N=2�u��gr��R%�������������:              �?        >G=��       �{�	�����A-*�
w
discriminator_loss*a	    'H�>    'H�>      �?!    'H�>) ��╅=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) b Q&J=2�u��gr��R%�������������:              �?        o~���       �{�	�����A2*�
w
discriminator_loss*a	   �X�>   �X�>      �?!   �X�>)���ɚ�=25�"�g��>G&�$�>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) ��)��O=2�MZ��K���u��gr���������:              �?        ��F��       �{�	}����A7*�
w
discriminator_loss*a	   �	x�>   �	x�>      �?!   �	x�>) �E�/Z=2��|�~�>���]���>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����)  "'&=2��ӤP�����z!�?���������:              �?        Nc@��       �{�	�R����A<*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>) Xݍ�=20�6�/n�>5�"�g��>�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) y@��5Q=2�MZ��K���u��gr���������:              �?         ���       �{�	�%����AA*�
w
discriminator_loss*a	   �З>   �З>      �?!   �З>) b�5M�A=2.��fc��>39W$:��>�������:              �?        
s
generator_loss*a	   �Д�   �Д�      �?!   �Д�) � �;=2X$�z��
�}�����������:              �?        +@��       �{�	9�����AF*�
w
discriminator_loss*a	   ` �>   ` �>      �?!   ` �>) ���&�C=2.��fc��>39W$:��>�������:              �?        
s
generator_loss*a	     ��     ��      �?!     ��)  � "=2��z!�?��T�L<���������:              �?        �	��       �{�	5�����AK*�
w
discriminator_loss*a	   @h�>   @h�>      �?!   @h�>) �AĮ�P=2�u��gr�>�MZ��K�>�������:              �?        
s
generator_loss*a	   � ���   � ���      �?!   � ���) �� Z =2��z!�?��T�L<���������:              �?        � Ӱ�       �{�	~�����AP*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�����d=2;9��R�>���?�ګ>�������:              �?        
s
generator_loss*a	   � @��   � @��      �?!   � @��) 	 ��0=2�4[_>������m!#���������:              �?        ���z�       �{�	}Z�����AU*�
w
discriminator_loss*a	   @ز>   @ز>      �?!   @ز>) a�ڀ1v=2��n����>�u`P+d�>�������:              �?        
s
generator_loss*a	   � ��   � ��      �?!   � ��) 2 ��@=2.��fc���X$�z���������:              �?        �NQ�