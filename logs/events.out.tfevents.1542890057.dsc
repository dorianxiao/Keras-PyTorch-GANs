       �K"	  @����Abrain.Event:2[<���     �;�	��y����A"��
r
Generator/noisePlaceholder*
shape:���������d*
dtype0*'
_output_shapes
:���������d
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
-Generator/first/Generator/firstleaky_relu/mulMul/Generator/first/Generator/firstleaky_relu/alpha6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
)Generator/first/Generator/firstleaky_reluMaximum-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
XGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0
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
`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformXGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shape*

seed*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
seed2*
dtype0* 
_output_shapes
:
��
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
RGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniformAddVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
�
7Generator/second/Generator/secondfully_connected/kernel
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
�
>Generator/second/Generator/secondfully_connected/kernel/AssignAssign7Generator/second/Generator/secondfully_connected/kernelRGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
<Generator/second/Generator/secondfully_connected/kernel/readIdentity7Generator/second/Generator/secondfully_connected/kernel*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
��
�
GGenerator/second/Generator/secondfully_connected/bias/Initializer/zerosConst*
_output_shapes	
:�*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB�*    *
dtype0
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
:Generator/second/Generator/secondfully_connected/bias/readIdentity5Generator/second/Generator/secondfully_connected/bias*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:�
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
VariableV2*
shared_name *P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
BGenerator/second/Generator/secondbatch_normalized/moving_mean/readIdentity=Generator/second/Generator/secondbatch_normalized/moving_mean*
_output_shapes	
:�*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean
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
FGenerator/second/Generator/secondbatch_normalized/moving_variance/readIdentityAGenerator/second/Generator/secondbatch_normalized/moving_variance*
_output_shapes	
:�*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance
�
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
?Generator/second/Generator/secondbatch_normalized/batchnorm/addAddFGenerator/second/Generator/secondbatch_normalized/moving_variance/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:�
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
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAdd?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
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
/Generator/second/Generator/secondleaky_relu/mulMul1Generator/second/Generator/secondleaky_relu/alphaAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
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

seed*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
seed2B*
dtype0* 
_output_shapes
:
��
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
<Generator/third/Generator/thirdfully_connected/kernel/AssignAssign5Generator/third/Generator/thirdfully_connected/kernelPGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
�
:Generator/third/Generator/thirdfully_connected/kernel/readIdentity5Generator/third/Generator/thirdfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:
��
�
EGenerator/third/Generator/thirdfully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB�*    
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
5Generator/third/Generator/thirdfully_connected/MatMulMatMul+Generator/second/Generator/secondleaky_relu:Generator/third/Generator/thirdfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
6Generator/third/Generator/thirdfully_connected/BiasAddBiasAdd5Generator/third/Generator/thirdfully_connected/MatMul8Generator/third/Generator/thirdfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
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
FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignAssign?Generator/third/Generator/thirdbatch_normalized/moving_variancePGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
DGenerator/third/Generator/thirdbatch_normalized/moving_variance/readIdentity?Generator/third/Generator/thirdbatch_normalized/moving_variance*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
_output_shapes	
:�*
T0
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
=Generator/third/Generator/thirdbatch_normalized/batchnorm/addAddDGenerator/third/Generator/thirdbatch_normalized/moving_variance/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/y*
_output_shapes	
:�*
T0
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/RsqrtRsqrt=Generator/third/Generator/thirdbatch_normalized/batchnorm/add*
_output_shapes	
:�*
T0
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
=Generator/third/Generator/thirdbatch_normalized/batchnorm/subSub9Generator/third/Generator/thirdbatch_normalized/beta/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1Add?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/sub*(
_output_shapes
:����������*
T0
t
/Generator/third/Generator/thirdleaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
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
UGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�
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
:Generator/forth/Generator/forthfully_connected/bias/AssignAssign3Generator/forth/Generator/forthfully_connected/biasEGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
�
8Generator/forth/Generator/forthfully_connected/bias/readIdentity3Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
	container 
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
`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB:�
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance
�
FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignAssign?Generator/forth/Generator/forthbatch_normalized/moving_variancePGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance
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
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAdd=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
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
7Generator/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0
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
5Generator/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *)
_class
loc:@Generator/dense/kernel*
valueB
 *z�k=
�
?Generator/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7Generator/dense/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@Generator/dense/kernel*
seed2�*
dtype0* 
_output_shapes
:
��*

seed
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
&Generator/dense/bias/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
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
Generator/dense/bias/AssignAssignGenerator/dense/bias&Generator/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:�
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
shape:����������*
dtype0*(
_output_shapes
:����������
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
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HY�=
�
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shape*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
seed2�*
dtype0* 
_output_shapes
:
��*

seed
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/subSub\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/max\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
_output_shapes
: *
T0
�
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mulMulfDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniform\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/sub*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��*
T0
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
BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignAssign;Discriminator/first/Discriminator/firstfully_connected/biasMDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
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
>Discriminator/first/Discriminator/firstfully_connected/BiasAddBiasAdd=Discriminator/first/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
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
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/subSub^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/max^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
_output_shapes
: *
T0
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
@Discriminator/second/Discriminator/secondfully_connected/BiasAddBiasAdd?Discriminator/second/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
~
9Discriminator/second/Discriminator/secondleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *��L>*
dtype0
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
9Discriminator/out/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*+
_class!
loc:@Discriminator/out/kernel*
valueB"      *
dtype0
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
7Discriminator/out/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv>
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
7Discriminator/out/kernel/Initializer/random_uniform/subSub7Discriminator/out/kernel/Initializer/random_uniform/max7Discriminator/out/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*+
_class!
loc:@Discriminator/out/kernel
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
Discriminator/out/kernel/AssignAssignDiscriminator/out/kernel3Discriminator/out/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	�
�
Discriminator/out/kernel/readIdentityDiscriminator/out/kernel*
_output_shapes
:	�*
T0*+
_class!
loc:@Discriminator/out/kernel
�
(Discriminator/out/bias/Initializer/zerosConst*
_output_shapes
:*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0
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
Discriminator/out/bias/AssignAssignDiscriminator/out/bias(Discriminator/out/bias/Initializer/zeros*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:*
use_locking(
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
Discriminator/out/BiasAddBiasAddDiscriminator/out/MatMulDiscriminator/out/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
q
Discriminator/out/SigmoidSigmoidDiscriminator/out/BiasAdd*
T0*'
_output_shapes
:���������
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
Discriminator/out_1/BiasAddBiasAddDiscriminator/out_1/MatMulDiscriminator/out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
u
Discriminator/out_1/SigmoidSigmoidDiscriminator/out_1/BiasAdd*
T0*'
_output_shapes
:���������
W
LogLogDiscriminator/out/Sigmoid*
T0*'
_output_shapes
:���������
J
sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
`
subSubsub/xDiscriminator/out_1/Sigmoid*'
_output_shapes
:���������*
T0
C
Log_1Logsub*'
_output_shapes
:���������*
T0
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
gradients/Mean_grad/ShapeShapeadd*
out_type0*
_output_shapes
:*
T0
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
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
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
gradients/add_grad/Sum_1Sumgradients/Mean_grad/truediv*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
gradients/sub_grad/Sum_1Sumgradients/Log_1_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
4gradients/Discriminator/out/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
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
@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyIdentity.gradients/Discriminator/out/MatMul_grad/MatMul9^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Discriminator/out/MatMul_grad/MatMul*(
_output_shapes
:����������
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
Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad
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
Fgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumSumIgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectXgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
2gradients/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
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
Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity2gradients/Discriminator/out_1/MatMul_grad/MatMul_1;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*E
_class;
97loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul_1
�
gradients/AddNAddNCgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1*
N*
_output_shapes
:*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad
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
\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/MulMul[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumSumJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul9Discriminator/second/Discriminator/secondleaky_relu/alpha[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Pgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
out_type0*
_output_shapes
:*
T0
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
Hgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumKgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectZgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
N*
_output_shapes
:	�*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1
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
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
gradients/AddN_3AddN_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N
�
]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
data_formatNHWC*
_output_shapes	
:�*
T0
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
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Shapeggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
�
Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosFillHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/Const*(
_output_shapes
:����������*
T0*

index_type0
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
[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1R^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1
�
Wgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMuljgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumSumHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulZgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1V^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
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
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumKgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1Zgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
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
agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Sgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
Ugradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
ggradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*h
_class^
\Zloc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_8AddNhgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:�*
T0
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
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
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
VariableV2*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
beta2_power/readIdentitybeta2_power*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: *
T0
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
TDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zerosFilldDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0
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
\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    
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
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/readIdentityDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
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
VariableV2*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
fDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0
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
VariableV2* 
_output_shapes
:
��*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
��*
dtype0
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
XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zerosFillhDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
��*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0
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
VariableV2*+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name 
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
"Discriminator/out/kernel/Adam/readIdentityDiscriminator/out/kernel/Adam*
_output_shapes
:	�*
T0*+
_class!
loc:@Discriminator/out/kernel
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
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	�
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
$Discriminator/out/bias/Adam_1/AssignAssignDiscriminator/out/bias/Adam_1/Discriminator/out/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(
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
Adam/beta2Adam/epsilongradients/AddN*
T0*)
_class
loc:@Discriminator/out/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(
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
gradients_1/Mean_1_grad/Shape_1ShapeLog_2*
_output_shapes
:*
T0*
out_type0
b
gradients_1/Mean_1_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_1/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*'
_output_shapes
:���������*
T0
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
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
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
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*'
_output_shapes
:���������
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
Egradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
�
Ggradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
2gradients_1/Discriminator/out_1/MatMul_grad/MatMulMatMulEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
Mgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Ogradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumOgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
�
Wgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpO^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeQ^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
�
_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeX^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*a
_classW
USloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape
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
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1bgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
lgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddNe^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
�
ngradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrade^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Ygradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMullgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:����������*
transpose_a( 
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
kgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityYgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMuld^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*l
_classb
`^loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
mgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1Identity[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1d^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*n
_classd
b`loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
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
Mgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeroskgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Hgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumKgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectZgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeHgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumMgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1\gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1Z^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*e
_class[
YWloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
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
lgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradc^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*p
_classf
dbloc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
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
kgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1b^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*l
_classb
`^loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
�
(gradients_1/Generator/Tanh_grad/TanhGradTanhGradGenerator/Tanhigradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/Generator/Tanh_grad/TanhGrad*
_output_shapes	
:�*
T0*
data_formatNHWC
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
.gradients_1/Generator/dense/MatMul_grad/MatMulMatMulAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyGenerator/dense/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
Cgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1SelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/SumSumAgradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectPgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulMulSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
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
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Hgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
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
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape
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
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:�*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1
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
agradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients_1/AddN_3AddNkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�
�
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_3:Generator/forth/Generator/forthbatch_normalized/gamma/read*
_output_shapes	
:�*
T0
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
Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
Cgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1SelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1SumCgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1Rgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
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
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumSumBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulTgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
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
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
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
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
�
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAddigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
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
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul
�
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�
�
Mgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/third/Generator/thirdfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Ogradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1MatMul+Generator/second/Generator/secondleaky_relu`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
��*
transpose_a(
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
agradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*b
_classX
VTloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1
�
gradients_1/AddN_5AddNkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:�*
T0
�
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_5:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:�
�
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_5?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt*
_output_shapes	
:�*
T0
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
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeShape/Generator/second/Generator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
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
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/zerosFillDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:����������*
T0
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
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
�
Vgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
Jgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1ReshapeFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
Tshape0*
_output_shapes	
:�*
T0
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
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
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
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Negb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:�*
T0
�
Ugradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Zgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOpl^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyV^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad
�
bgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitykgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape
�
dgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
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
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Muld^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:�
�
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*k
_classa
_]loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:�*
T0
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
cgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1Z^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*d
_classZ
XVloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��*
T0
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
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mulb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:�*
T0
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeShape-Generator/first/Generator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
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
Hgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
Ogradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	d�*
transpose_a(*
transpose_b( *
T0
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
VariableV2*'
_class
loc:@Generator/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
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
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@Generator/dense/bias
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
RGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    
�
LGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zerosFill\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	d�*
T0
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
^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d   �   *
dtype0*
_output_shapes
:
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
NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	d�
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
?Generator/first/Generator/firstfully_connected/bias/Adam/AssignAssign8Generator/first/Generator/firstfully_connected/bias/AdamJGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
NGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zerosFill^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorTGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
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
`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0*
_output_shapes
:
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
EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignAssign>Generator/second/Generator/secondfully_connected/kernel/Adam_1PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(
�
CGenerator/second/Generator/secondfully_connected/kernel/Adam_1/readIdentity>Generator/second/Generator/secondfully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
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
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container 
�
AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAssign:Generator/second/Generator/secondfully_connected/bias/AdamLGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(
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
CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignAssign<Generator/second/Generator/secondfully_connected/bias/Adam_1NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
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
AGenerator/second/Generator/secondbatch_normalized/gamma/Adam/readIdentity<Generator/second/Generator/secondbatch_normalized/gamma/Adam*
_output_shapes	
:�*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma
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
VariableV2*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/readIdentity>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:�*
T0
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
BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignAssign;Generator/second/Generator/secondbatch_normalized/beta/AdamMGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
@Generator/second/Generator/secondbatch_normalized/beta/Adam/readIdentity;Generator/second/Generator/secondbatch_normalized/beta/Adam*
_output_shapes	
:�*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
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
DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignAssign=Generator/second/Generator/secondbatch_normalized/beta/Adam_1OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
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
VariableV2*
shape:
��*
dtype0* 
_output_shapes
:
��*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container 
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
AGenerator/third/Generator/thirdfully_connected/bias/Adam_1/AssignAssign:Generator/third/Generator/thirdfully_connected/bias/Adam_1LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAssign:Generator/third/Generator/thirdbatch_normalized/gamma/AdamLGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignAssign<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�
�
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/readIdentity<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:�*
T0
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
BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignAssign;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
@Generator/third/Generator/thirdbatch_normalized/beta/Adam_1/readIdentity;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:�*
T0
�
\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0
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
TGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0
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
ZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:�
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:�
�
?Generator/forth/Generator/forthfully_connected/bias/Adam/AssignAssign8Generator/forth/Generator/forthfully_connected/bias/AdamJGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(
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
LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:�
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
AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAssign:Generator/forth/Generator/forthfully_connected/bias/Adam_1LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�
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
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAssign:Generator/forth/Generator/forthbatch_normalized/gamma/AdamLGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
?Generator/forth/Generator/forthbatch_normalized/gamma/Adam/readIdentity:Generator/forth/Generator/forthbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:�
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
NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/Const*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0
�
<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1
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
CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignAssign<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(
�
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/readIdentity<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:�
�
[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:�*
dtype0
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
@Generator/forth/Generator/forthbatch_normalized/beta/Adam/AssignAssign9Generator/forth/Generator/forthbatch_normalized/beta/AdamKGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
>Generator/forth/Generator/forthbatch_normalized/beta/Adam/readIdentity9Generator/forth/Generator/forthbatch_normalized/beta/Adam*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:�
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
BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignAssign;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
VariableV2*'
_class
loc:@Generator/dense/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
KAdam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdam	ApplyAdam3Generator/first/Generator/firstfully_connected/bias8Generator/first/Generator/firstfully_connected/bias/Adam:Generator/first/Generator/firstfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
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
MAdam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdam	ApplyAdam5Generator/second/Generator/secondfully_connected/bias:Generator/second/Generator/secondfully_connected/bias/Adam<Generator/second/Generator/secondfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
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
MAdam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdbatch_normalized/gamma:Generator/third/Generator/thirdbatch_normalized/gamma/Adam<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:�
�
LAdam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/third/Generator/thirdbatch_normalized/beta9Generator/third/Generator/thirdbatch_normalized/beta/Adam;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
_output_shapes	
:�*
use_locking( *
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
use_nesterov( 
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
KAdam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdam	ApplyAdam3Generator/forth/Generator/forthfully_connected/bias8Generator/forth/Generator/forthfully_connected/bias/Adam:Generator/forth/Generator/forthfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( 
�
MAdam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthbatch_normalized/gamma:Generator/forth/Generator/forthbatch_normalized/gamma/Adam<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
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
,Adam_1/update_Generator/dense/bias/ApplyAdam	ApplyAdamGenerator/dense/biasGenerator/dense/bias/AdamGenerator/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias
�


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
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
: "��ٰb     ��x6	������AJ��
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
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
: *
T0
�
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�
�
PGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniformAddTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
:	d�*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
�
5Generator/first/Generator/firstfully_connected/kernel
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
VariableV2*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:�*
dtype0
�
:Generator/first/Generator/firstfully_connected/bias/AssignAssign3Generator/first/Generator/firstfully_connected/biasEGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
8Generator/first/Generator/firstfully_connected/bias/readIdentity3Generator/first/Generator/firstfully_connected/bias*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:�
�
5Generator/first/Generator/firstfully_connected/MatMulMatMulGenerator/noise:Generator/first/Generator/firstfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
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
-Generator/first/Generator/firstleaky_relu/mulMul/Generator/first/Generator/firstleaky_relu/alpha6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
)Generator/first/Generator/firstleaky_reluMaximum-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
�
XGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"�      *
dtype0
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
`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformXGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shape*
seed2*
dtype0* 
_output_shapes
:
��*

seed*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
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
RGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniformAddVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
��*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
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
7Generator/second/Generator/secondfully_connected/MatMulMatMul)Generator/first/Generator/firstleaky_relu<Generator/second/Generator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
8Generator/second/Generator/secondfully_connected/BiasAddBiasAdd7Generator/second/Generator/secondfully_connected/MatMul:Generator/second/Generator/secondfully_connected/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
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
VariableV2*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
<Generator/second/Generator/secondbatch_normalized/gamma/readIdentity7Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:�*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma
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
VariableV2*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
;Generator/second/Generator/secondbatch_normalized/beta/readIdentity6Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:�*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
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
VariableV2*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
	container *
shape:�*
dtype0
�
DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignAssign=Generator/second/Generator/secondbatch_normalized/moving_meanOGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
validate_shape(
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
HGenerator/second/Generator/secondbatch_normalized/moving_variance/AssignAssignAGenerator/second/Generator/secondbatch_normalized/moving_varianceRGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance
�
FGenerator/second/Generator/secondbatch_normalized/moving_variance/readIdentityAGenerator/second/Generator/secondbatch_normalized/moving_variance*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
_output_shapes	
:�*
T0
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
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAdd?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:����������
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
seed2B*
dtype0* 
_output_shapes
:
��*

seed*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
�
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
_output_shapes
: 
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
<Generator/third/Generator/thirdfully_connected/kernel/AssignAssign5Generator/third/Generator/thirdfully_connected/kernelPGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
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
5Generator/third/Generator/thirdfully_connected/MatMulMatMul+Generator/second/Generator/secondleaky_relu:Generator/third/Generator/thirdfully_connected/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
6Generator/third/Generator/thirdfully_connected/BiasAddBiasAdd5Generator/third/Generator/thirdfully_connected/MatMul8Generator/third/Generator/thirdfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
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
VariableV2*
_output_shapes	
:�*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:�*
dtype0
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
:Generator/third/Generator/thirdbatch_normalized/gamma/readIdentity5Generator/third/Generator/thirdbatch_normalized/gamma*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:�*
T0
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
;Generator/third/Generator/thirdbatch_normalized/beta/AssignAssign4Generator/third/Generator/thirdbatch_normalized/betaFGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
9Generator/third/Generator/thirdbatch_normalized/beta/readIdentity4Generator/third/Generator/thirdbatch_normalized/beta*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:�*
T0
�
MGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
valueB�*    
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
-Generator/third/Generator/thirdleaky_relu/mulMul/Generator/third/Generator/thirdleaky_relu/alpha?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
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
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  �=*
dtype0
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
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/sub*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
��*
T0
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
<Generator/forth/Generator/forthfully_connected/kernel/AssignAssign5Generator/forth/Generator/forthfully_connected/kernelPGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
��*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(
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
EGenerator/forth/Generator/forthfully_connected/bias/Initializer/zerosFillUGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorKGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/Const*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0
�
3Generator/forth/Generator/forthfully_connected/bias
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
:Generator/forth/Generator/forthfully_connected/bias/AssignAssign3Generator/forth/Generator/forthfully_connected/biasEGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
6Generator/forth/Generator/forthfully_connected/BiasAddBiasAdd5Generator/forth/Generator/forthfully_connected/MatMul8Generator/forth/Generator/forthfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
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
<Generator/forth/Generator/forthbatch_normalized/gamma/AssignAssign5Generator/forth/Generator/forthbatch_normalized/gammaFGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
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
SGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB
 *    
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
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean
�
BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignAssign;Generator/forth/Generator/forthbatch_normalized/moving_meanMGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
validate_shape(
�
@Generator/forth/Generator/forthbatch_normalized/moving_mean/readIdentity;Generator/forth/Generator/forthbatch_normalized/moving_mean*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
_output_shapes	
:�*
T0
�
`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB:�
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
=Generator/forth/Generator/forthbatch_normalized/batchnorm/mulMul?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt:Generator/forth/Generator/forthbatch_normalized/gamma/read*
_output_shapes	
:�*
T0
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAdd=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*(
_output_shapes
:����������*
T0
�
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2Mul@Generator/forth/Generator/forthbatch_normalized/moving_mean/read=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:�
�
=Generator/forth/Generator/forthbatch_normalized/batchnorm/subSub9Generator/forth/Generator/forthbatch_normalized/beta/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2*
_output_shapes	
:�*
T0
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
5Generator/dense/kernel/Initializer/random_uniform/subSub5Generator/dense/kernel/Initializer/random_uniform/max5Generator/dense/kernel/Initializer/random_uniform/min*)
_class
loc:@Generator/dense/kernel*
_output_shapes
: *
T0
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
&Generator/dense/bias/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueB�*    *
dtype0*
_output_shapes	
:�
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
Generator/dense/bias/readIdentityGenerator/dense/bias*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:�
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
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shape*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
seed2�*
dtype0* 
_output_shapes
:
��*

seed
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
5Discriminator/first/Discriminator/firstleaky_relu/mulMul7Discriminator/first/Discriminator/firstleaky_relu/alpha>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
1Discriminator/first/Discriminator/firstleaky_reluMaximum5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:����������*
T0
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
ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniformAdd^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mul^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
��
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
VariableV2*
_output_shapes	
:�*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:�*
dtype0
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
9Discriminator/second/Discriminator/secondleaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *��L>
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
7Discriminator/out/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv�*
dtype0
�
7Discriminator/out/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv>
�
ADiscriminator/out/kernel/Initializer/random_uniform/RandomUniformRandomUniform9Discriminator/out/kernel/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes
:	�*

seed*
T0*+
_class!
loc:@Discriminator/out/kernel
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
Discriminator/out/kernel/AssignAssignDiscriminator/out/kernel3Discriminator/out/kernel/Initializer/random_uniform*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0
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
VariableV2*)
_class
loc:@Discriminator/out/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
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
Discriminator/out/BiasAddBiasAddDiscriminator/out/MatMulDiscriminator/out/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
q
Discriminator/out/SigmoidSigmoidDiscriminator/out/BiasAdd*
T0*'
_output_shapes
:���������
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
3Discriminator/first_1/Discriminator/firstleaky_reluMaximum7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
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
Discriminator/out_1/BiasAddBiasAddDiscriminator/out_1/MatMulDiscriminator/out/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
u
Discriminator/out_1/SigmoidSigmoidDiscriminator/out_1/BiasAdd*
T0*'
_output_shapes
:���������
W
LogLogDiscriminator/out/Sigmoid*
T0*'
_output_shapes
:���������
J
sub/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
ConstConst*
_output_shapes
:*
valueB"       *
dtype0
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
sub_1/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
Const_1Const*
_output_shapes
:*
valueB"       *
dtype0
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
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
\
gradients/Mean_grad/ShapeShapeadd*
_output_shapes
:*
T0*
out_type0
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
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
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
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
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
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0*
Truncate( 
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
gradients/add_grad/Shape_1ShapeLog_1*
out_type0*
_output_shapes
:*
T0
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
gradients/add_grad/Sum_1Sumgradients/Mean_grad/truediv*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
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
ReciprocalDiscriminator/out/Sigmoid,^gradients/add_grad/tuple/control_dependency*'
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
Reciprocalsub.^gradients/add_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
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
4gradients/Discriminator/out/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
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
@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyIdentity.gradients/Discriminator/out/MatMul_grad/MatMul9^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/Discriminator/out/MatMul_grad/MatMul*(
_output_shapes
:����������
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
Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad
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
Igradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectSelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqual@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Kgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1SelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Fgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumSumIgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectXgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
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
[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeT^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:����������
�
]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1T^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1
�
0gradients/Discriminator/out_1/MatMul_grad/MatMulMatMulCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
Bgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity0gradients/Discriminator/out_1/MatMul_grad/MatMul;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:����������
�
Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity2gradients/Discriminator/out_1/MatMul_grad/MatMul_1;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*E
_class;
97loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	�*
T0
�
gradients/AddNAddNCgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:*
T0
�
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
�
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/MulMul[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:����������
�
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumSumJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:
�
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
Hgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumKgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectZgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
gradients/AddN_1AddNBgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
N*
_output_shapes
:	�
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
hgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_2a^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:����������
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
Rgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
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
cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1Z^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*e
_class[
YWloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
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
igradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
gradients/AddN_3AddN_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:����������*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
�
]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
data_formatNHWC*
_output_shapes	
:�*
T0
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
Igradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1SelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Dgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumSumGgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectVgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1R^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1
�
Wgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMuljgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Ygradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relujgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
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
Zgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
]gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeV^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*_
_classU
SQloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape
�
_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1V^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
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
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
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
[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeT^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*]
_classS
QOloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape
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
N*(
_output_shapes
:����������*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1
�
Ygradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
_output_shapes	
:�*
T0*
data_formatNHWC
�
^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6Z^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
�
fgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
�
hgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
�
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
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
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
_gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
�
agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������
�
Sgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
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
gradients/AddN_7AddN]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:����������*
T0
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
jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*n
_classd
b`loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�*
T0
�
Ugradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
Wgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
��*
transpose_a(*
transpose_b( *
T0
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
beta1_power/initial_valueConst*
_output_shapes
: *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *fff?*
dtype0
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
TDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zerosFilldDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
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
KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignAssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/readIdentityDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
��
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
VariableV2*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
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
TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:�*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB�*    *
dtype0
�
BDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1
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
\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
VDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zerosFillfDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensor\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
��*
T0
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
KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamVDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
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
VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB�*    
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
IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1*
_output_shapes	
:�*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias
�
/Discriminator/out/kernel/Adam/Initializer/zerosConst*
_output_shapes
:	�*+
_class!
loc:@Discriminator/out/kernel*
valueB	�*    *
dtype0
�
Discriminator/out/kernel/Adam
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
$Discriminator/out/kernel/Adam/AssignAssignDiscriminator/out/kernel/Adam/Discriminator/out/kernel/Adam/Initializer/zeros*
_output_shapes
:	�*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(
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
"Discriminator/out/bias/Adam/AssignAssignDiscriminator/out/bias/Adam-Discriminator/out/bias/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:
�
 Discriminator/out/bias/Adam/readIdentityDiscriminator/out/bias/Adam*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
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
VariableV2*)
_class
loc:@Discriminator/out/bias*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
$Discriminator/out/bias/Adam_1/AssignAssignDiscriminator/out/bias/Adam_1/Discriminator/out/bias/Adam_1/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(
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

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
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
Adam/beta2Adam/epsilongradients/AddN*
_output_shapes
:*
use_locking( *
T0*)
_class
loc:@Discriminator/out/bias*
use_nesterov( 
�
Adam/mulMulbeta1_power/read
Adam/beta1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
�
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
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
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
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
gradients_1/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
_output_shapes
: *
T0
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
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
gradients_1/sub_1_grad/Sum_1Sumgradients_1/Log_2_grad/mul.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid1gradients_1/sub_1_grad/tuple/control_dependency_1*'
_output_shapes
:���������*
T0
�
8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
_output_shapes
:*
T0*
data_formatNHWC
�
=gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp9^gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad9^gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
�
Egradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
�
Ggradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
2gradients_1/Discriminator/out_1/MatMul_grad/MatMulMatMulEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b(*
T0
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
Dgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/Discriminator/out_1/MatMul_grad/MatMul=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*E
_class;
97loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:����������*
T0
�
Fgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*G
_class=
;9loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	�*
T0
�
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
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
\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
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
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
mgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1Identity[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1d^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*n
_classd
b`loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
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
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
Hgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumKgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectZgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeHgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
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
]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeV^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*_
_classU
SQloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:����������*
T0
�
_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1V^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
�
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
jgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1c^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:����������*
T0
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
igradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulb^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*j
_class`
^\loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul
�
kgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1b^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
��
�
(gradients_1/Generator/Tanh_grad/TanhGradTanhGradGenerator/Tanhigradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
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
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zerosFillBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:����������
�
Ggradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqualGreaterEqual-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:����������*
T0
�
Pgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Agradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectSelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros*(
_output_shapes
:����������*
T0
�
Cgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1SelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/SumSumAgradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectPgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Dgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
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
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
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
N*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
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
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_2fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAddigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
�
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
_output_shapes	
:�*
T0*
Tshape0
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
Sgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
_output_shapes	
:�*
T0*
data_formatNHWC
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
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*i
_class_
][loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
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
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:�*
T0
�
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1
�
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeShape-Generator/third/Generator/thirdleaky_relu/mul*
out_type0*
_output_shapes
:*
T0
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
Agradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectSelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
�
Cgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1SelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:����������*
T0
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
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1SumCgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1Rgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
�
Kgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeE^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1
�
Sgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeL^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*U
_classK
IGloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape
�
Ugradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1L^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1
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
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulMulSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumSumBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulTgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Mul/Generator/third/Generator/thirdleaky_relu/alphaSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:����������
�
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
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
Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*[
_classQ
OMloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:����������*
T0
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
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
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
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/third/Generator/thirdfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
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
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape*(
_output_shapes
:����������*
T0*
Tshape0
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
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:�*
T0
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
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:�*
T0
�
Sgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:�*
T0
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
Igradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualGreaterEqual/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Rgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
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
@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumSumCgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectRgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
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
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulMulUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:����������
�
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumSumDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulVgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeReshapeDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
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
Jgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1ReshapeFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1R^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*]
_classS
QOloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1
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
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:�
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
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:�*
T0
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
bgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitykgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:����������*
T0
�
dgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
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
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mulb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:�*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul
�
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:�
�
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeShape-Generator/first/Generator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
�
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Shapeagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
out_type0*
_output_shapes
:*
T0
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
Agradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectSelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:����������
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
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
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
Hgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*(
_output_shapes
:����������*
T0*
Tshape0
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
gradients_1/AddN_8AddNUgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:����������*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*
N
�
Sgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
data_formatNHWC*
_output_shapes	
:�*
T0
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
Mgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/first/Generator/firstfully_connected/kernel/read*
transpose_b(*
T0*'
_output_shapes
:���������d*
transpose_a( 
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
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
w
beta1_power_1/readIdentitybeta1_power_1*'
_class
loc:@Generator/dense/bias*
_output_shapes
: *
T0
�
beta2_power_1/initial_valueConst*
_output_shapes
: *'
_class
loc:@Generator/dense/bias*
valueB
 *w�?*
dtype0
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
AGenerator/first/Generator/firstfully_connected/kernel/Adam/AssignAssign:Generator/first/Generator/firstfully_connected/kernel/AdamLGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d�*
use_locking(
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
CGenerator/first/Generator/firstfully_connected/kernel/Adam_1/AssignAssign<Generator/first/Generator/firstfully_connected/kernel/Adam_1NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d�
�
AGenerator/first/Generator/firstfully_connected/kernel/Adam_1/readIdentity<Generator/first/Generator/firstfully_connected/kernel/Adam_1*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d�*
T0
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
?Generator/first/Generator/firstfully_connected/bias/Adam/AssignAssign8Generator/first/Generator/firstfully_connected/bias/AdamJGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
�
=Generator/first/Generator/firstfully_connected/bias/Adam/readIdentity8Generator/first/Generator/firstfully_connected/bias/Adam*
_output_shapes	
:�*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
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
AGenerator/first/Generator/firstfully_connected/bias/Adam_1/AssignAssign:Generator/first/Generator/firstfully_connected/bias/Adam_1LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zeros*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignAssign<Generator/second/Generator/secondfully_connected/kernel/AdamNGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
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
AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAssign:Generator/second/Generator/secondfully_connected/bias/AdamLGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
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
CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignAssign<Generator/second/Generator/secondfully_connected/bias/Adam_1NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:�
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
MGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB�*    
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
BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignAssign;Generator/second/Generator/secondbatch_normalized/beta/AdamMGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�
�
@Generator/second/Generator/secondbatch_normalized/beta/Adam/readIdentity;Generator/second/Generator/secondbatch_normalized/beta/Adam*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:�
�
OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB�*    
�
=Generator/second/Generator/secondbatch_normalized/beta/Adam_1
VariableV2*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignAssign=Generator/second/Generator/secondbatch_normalized/beta/Adam_1OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
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
?Generator/third/Generator/thirdfully_connected/kernel/Adam/readIdentity:Generator/third/Generator/thirdfully_connected/kernel/Adam* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
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
CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignAssign<Generator/third/Generator/thirdfully_connected/kernel/Adam_1NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0
�
AGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/readIdentity<Generator/third/Generator/thirdfully_connected/kernel/Adam_1* 
_output_shapes
:
��*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
�
JGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB�*    
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
KGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:�*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB�*    
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
PGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    
�
JGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zerosFillZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorPGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/Const*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:�*
T0
�
8Generator/forth/Generator/forthfully_connected/bias/Adam
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
?Generator/forth/Generator/forthfully_connected/bias/Adam/AssignAssign8Generator/forth/Generator/forthfully_connected/bias/AdamJGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:�
�
:Generator/forth/Generator/forthfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:�
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
RGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0
�
LGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zerosFill\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/Const*
_output_shapes	
:�*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0
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
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAssign:Generator/forth/Generator/forthbatch_normalized/gamma/AdamLGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignAssign<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
@Generator/forth/Generator/forthbatch_normalized/beta/Adam/AssignAssign9Generator/forth/Generator/forthbatch_normalized/beta/AdamKGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
>Generator/forth/Generator/forthbatch_normalized/beta/Adam/readIdentity9Generator/forth/Generator/forthbatch_normalized/beta/Adam*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:�
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
BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignAssign;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
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
"Generator/dense/kernel/Adam/AssignAssignGenerator/dense/kernel/Adam-Generator/dense/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
��*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel
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
VariableV2*'
_class
loc:@Generator/dense/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
"Generator/dense/bias/Adam_1/AssignAssignGenerator/dense/bias/Adam_1-Generator/dense/bias/Adam_1/Initializer/zeros*
_output_shapes	
:�*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(
�
 Generator/dense/bias/Adam_1/readIdentityGenerator/dense/bias/Adam_1*
T0*'
_class
loc:@Generator/dense/bias*
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
MAdam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/first/Generator/firstfully_connected/kernel:Generator/first/Generator/firstfully_connected/kernel/Adam<Generator/first/Generator/firstfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
use_nesterov( *
_output_shapes
:	d�
�
KAdam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdam	ApplyAdam3Generator/first/Generator/firstfully_connected/bias8Generator/first/Generator/firstfully_connected/bias/Adam:Generator/first/Generator/firstfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
OAdam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdam	ApplyAdam7Generator/second/Generator/secondfully_connected/kernel<Generator/second/Generator/secondfully_connected/kernel/Adam>Generator/second/Generator/secondfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( *
T0
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
NAdam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdam	ApplyAdam6Generator/second/Generator/secondbatch_normalized/beta;Generator/second/Generator/secondbatch_normalized/beta/Adam=Generator/second/Generator/secondbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0
�
MAdam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdfully_connected/kernel:Generator/third/Generator/thirdfully_connected/kernel/Adam<Generator/third/Generator/thirdfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
��*
use_locking( 
�
KAdam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdam	ApplyAdam3Generator/third/Generator/thirdfully_connected/bias8Generator/third/Generator/thirdfully_connected/bias/Adam:Generator/third/Generator/thirdfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:�*
use_locking( *
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
use_nesterov( 
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
LAdam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/forth/Generator/forthbatch_normalized/beta9Generator/forth/Generator/forthbatch_normalized/beta/Adam;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
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
,Adam_1/update_Generator/dense/bias/ApplyAdam	ApplyAdamGenerator/dense/biasGenerator/dense/bias/AdamGenerator/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
use_nesterov( *
_output_shapes	
:�
�


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
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
: ""
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
Discriminator/out/bias:0Discriminator/out/bias/AssignDiscriminator/out/bias/read:02*Discriminator/out/bias/Initializer/zeros:08yU���       �N�	w`
����A*�
w
discriminator_loss*a	    m#�?    m#�?      �?!    m#�?) H��� @23?��|�?�E̟���?�������:              �?        
s
generator_loss*a	    ���    ���      �?!    ���) @KG��?2uo�p�+Se*8��������:              �?        ��	��       �{�	��,����A*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) B)��	@2S�Fi��?ܔ�.�u�?�������:              �?        
s
generator_loss*a	   @S���   @S���      �?!   @S���)����@2yL�������E̟�����������:              �?        RqpD�       �{�	�p4����A
*�
w
discriminator_loss*a	    �@    �@      �?!    �@)@�,��G@2w`<f@�6v��@�������:              �?        
s
generator_loss*a	   �"��   �"��      �?!   �"��) d��U}@2w`<f���tM��������:              �?        �<p(�       �{�	Ss;����A*�
w
discriminator_loss*a	   �"@@   �"@@      �?!   �"@@) dJdJ�@2ܔ�.�u�?��tM@�������:              �?        
s
generator_loss*a	   ��K �   ��K �      �?!   ��K �) !��ݘ@2��tM�ܔ�.�u���������:              �?        <����       �{�	nB����A*�
w
discriminator_loss*a	   ����?   ����?      �?!   ����?) �؝�X�?2\l�9�?+Se*8�?�������:              �?        
s
generator_loss*a	   ��޿   ��޿      �?!   ��޿) �O�i��?2�1%���Z%�޿�������:              �?        7�h�       �{�	�zI����A*�
w
discriminator_loss*a	    1��?    1��?      �?!    1��?) ��pԯ?2�Z�_���?����?�������:              �?        
s
generator_loss*a	   ��Ҩ�   ��Ҩ�      �?!   ��Ҩ�) 2	8^Ac?2���g�骿�g���w���������:              �?        X�^H�       �{�	��P����A*�
w
discriminator_loss*a	   �wz�?   �wz�?      �?!   �wz�?)� ��~��?2�@�"��?�K?�?�������:              �?        
s
generator_loss*a	   @%���   @%���      �?!   @%���)�\���I-?2�#�h/���7c_XY���������:              �?        ���       �{�	�~W����A#*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)�P���ɏ?2!�����?Ӗ8��s�?�������:              �?        
s
generator_loss*a	   ����   ����      �?!   ����) r����C?2}Y�4j���"�uԖ��������:              �?        �Φ��       �{�	�t^����A(*�
w
discriminator_loss*a	   �_��?   �_��?      �?!   �_��?) �mƕ?2Ӗ8��s�?�?>8s2�?�������:              �?        
s
generator_loss*a	    ����    ����      �?!    ����) @N>�x?2� l(����{ �ǳ���������:              �?        |�k�       �{�	��~����A-*�
w
discriminator_loss*a	   ��<�?   ��<�?      �?!   ��<�?)����K~�?2��(!�ؼ?!�����?�������:              �?        
s
generator_loss*a	   @P��   @P��      �?!   @P��)�8A;ԣn?2����iH��I�����������:              �?        �@� �       �{�	Cu�����A2*�
w
discriminator_loss*a	   �4�?   �4�?      �?!   �4�?) �k��hp?2I���?����iH�?�������:              �?        
s
generator_loss*a	   @^rx�   @^rx�      �?!   @^rx�)��� &�?2o��5sz�*QH�x��������:              �?        zt�       �{�	�������A7*�
w
discriminator_loss*a	   @�/�?   @�/�?      �?!   @�/�?)����en?2I���?����iH�?�������:              �?        
s
generator_loss*a	   �FL�   �FL�      �?!   �FL�) 7|���>2IcD���L��qU���I��������:              �?        ?(q�       �{�	*������A<*�
w
discriminator_loss*a	   ��[�?   ��[�?      �?!   ��[�?) �*Z��\?2`��a�8�?�/�*>�?�������:              �?        
s
generator_loss*a	    �2�    �2�      �?!    �2�) @�`v>2�u�w74���82��������:              �?        �˹{�       �{�	������AA*�
w
discriminator_loss*a	   @�,�?   @�,�?      �?!   @�,�?)�T�I��@?2�"�uԖ?}Y�4j�?�������:              �?        
s
generator_loss*a	   @�,�   @�,�      �?!   @�,�)�L�{�h>2��VlQ.��7Kaa+��������:              �?        &��[�       �{�	������AF*�
w
discriminator_loss*a	    6�?    6�?      �?!    6�?) @��u�;?2^�S���?�"�uԖ?�������:              �?        
s
generator_loss*a	   ฌ(�   ฌ(�      �?!   ฌ(�) �%��b>2I�I�)�(�+A�F�&��������:              �?        [D�       �{�	�������AK*�
w
discriminator_loss*a	    �3�?    �3�?      �?!    �3�?) ��x��(?2#�+(�ŉ?�7c_XY�?�������:              �?        
s
generator_loss*a	   �wu2�   �wu2�      �?!   �wu2�) �,G�Ku>2�u�w74���82��������:              �?        (@\I�       �{�	������AP*�
w
discriminator_loss*a	   ���{?   ���{?      �?!   ���{?) D���j?2o��5sz?���T}?�������:              �?        
s
generator_loss*a	   �]�9�   �]�9�      �?!   �]�9�) d�Z�`�>2��%>��:�uܬ�@8��������:              �?        ��7��       �{�	)Hؓ���AU*�
w
discriminator_loss*a	   `l�?   `l�?      �?!   `l�?) 	E��?2���T}?>	� �?�������:              �?        
s
generator_loss*a	   �}F�   �}F�      �?!   �}F�) dX�]W�>2
����G�a�$��{E��������:              �?        ��`�       �{�	������AZ*�
w
discriminator_loss*a	    z�?    z�?      �?!    z�?) ���?2���T}?>	� �?�������:              �?        
s
generator_loss*a	   �SO�   �SO�      �?!   �SO�) ��[	=�>2k�1^�sO�IcD���L��������:              �?        ,���       �{�	�瓨��A_*�
w
discriminator_loss*a	    @sy?    @sy?      �?!    @sy?)  ��=?2*QH�x?o��5sz?�������:              �?        
s
generator_loss*a	    �V�    �V�      �?!    �V�) ���[�>2ܗ�SsW�<DKc��T��������:              �?        �R���       �{�	-��Ad*�
w
discriminator_loss*a	   ��=x?   ��=x?      �?!   ��=x?) �F�\?2*QH�x?o��5sz?�������:              �?        
s
generator_loss*a	   @8ha�   @8ha�      �?!   @8ha�) �d��>2���%��b��l�P�`��������:              �?        ��i��       �{�	&�����Ai*�
w
discriminator_loss*a	    ��y?    ��y?      �?!    ��y?) ��n��?2*QH�x?o��5sz?�������:              �?        
s
generator_loss*a	   �Np�   �Np�      �?!   �Np�) ��m�0�>2;8�clp��N�W�m��������:              �?        �=F��       �{�	������An*�
w
discriminator_loss*a	   ��\�?   ��\�?      �?!   ��\�?) �%�?e'?2#�+(�ŉ?�7c_XY�?�������:              �?        
s
generator_loss*a	    �P~�    �P~�      �?!    �P~�) �X�w�?2>	� �����T}��������:              �?        6RW��       �{�	�o����As*�
w
discriminator_loss*a	   @�p�?   @�p�?      �?!   @�p�?) �JR�3?2���&�?�Rc�ݒ?�������:              �?        
s
generator_loss*a	   `�L��   `�L��      �?!   `�L��) �@؝%?2�7c_XY��#�+(�ŉ��������:              �?        at�`�       �{�	�z
����Ax*�
w
discriminator_loss*a	   �(�?   �(�?      �?!   �(�?) ɟj,2?2�#�h/�?���&�?�������:              �?        
s
generator_loss*a	   �B���   �B���      �?!   �B���) ��� �!?2#�+(�ŉ�eiS�m���������:              �?        �-�g�       �{�	_�7����A}*�
w
discriminator_loss*a	   ��z�?   ��z�?      �?!   ��z�?) Tq�6:?2�Rc�ݒ?^�S���?�������:              �?        
s
generator_loss*a	   ���x�   ���x�      �?!   ���x�)����}D?2o��5sz�*QH�x��������:              �?        st�\�       b�D�	�?����A�*�
w
discriminator_loss*a	   �]M�?   �]M�?      �?!   �]M�?) Q}�1�0?2�#�h/�?���&�?�������:              �?        
s
generator_loss*a	   `�:f�   `�:f�      �?!   `�:f�)@��h��>2Tw��Nof�5Ucv0ed��������:              �?        {���       b�D�	�PF����A�*�
w
discriminator_loss*a	    ;�?    ;�?      �?!    ;�?)  ���X"?2eiS�m�?#�+(�ŉ?�������:              �?        
s
generator_loss*a	   �|b�   �|b�      �?!   �|b�)@:M��Z�>2���%��b��l�P�`��������:              �?        :dϮ�       b�D�	w�M����A�*�
w
discriminator_loss*a	   @g�?   @g�?      �?!   @g�?)�$A�J(?2#�+(�ŉ?�7c_XY�?�������:              �?        
s
generator_loss*a	   ��j�   ��j�      �?!   ��j�) �?#,J�>2ߤ�(g%k�P}���h��������:              �?        :���       b�D�	~|T����A�*�
w
discriminator_loss*a	   ����?   ����?      �?!   ����?) �m獭5?2���&�?�Rc�ݒ?�������:              �?        
s
generator_loss*a	    j}�    j}�      �?!    j}�) �u��y
?2>	� �����T}��������:              �?        ޭ��       b�D�	�u[����A�*�
w
discriminator_loss*a	   ��F�?   ��F�?      �?!   ��F�?) �_�+�@?2�"�uԖ?}Y�4j�?�������:              �?        
s
generator_loss*a	   ��D��   ��D��      �?!   ��D��) ��g4?2���J�\������=����������:              �?        �5i�       b�D�	Jgb����A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) �,��F?2}Y�4j�?��<�A��?�������:              �?        
s
generator_loss*a	   @}��   @}��      �?!   @}��) ��]?2���J�\������=����������:              �?         ��3�       b�D�	�]i����A�*�
w
discriminator_loss*a	   �FĢ?   �FĢ?      �?!   �FĢ?) �v`V?2�uS��a�?`��a�8�?�������:              �?        
s
generator_loss*a	    ����    ����      �?!    ����)  ^A/@?2�"�uԖ�^�S�����������:              �?        N^ ��       b�D�	������A�*�
w
discriminator_loss*a	    -8�?    -8�?      �?!    -8�?)@D_ %��?2����?_&A�o��?�������:              �?        
s
generator_loss*a	   �7�̿   �7�̿      �?!   �7�̿) �V�k��?2�Z�_��ο�K?̿�������:              �?        ��QB�       b�D�	������A�*�
w
discriminator_loss*a	   �v�@   �v�@      �?!   �v�@)���~EJ-@2u�rʭ�@�DK��@�������:              �?        
s
generator_loss*a	   `��   `��      �?!   `��) 7hT�&@2u�rʭ���Š)U	��������:              �?        ����       b�D�	s�����A�*�
w
discriminator_loss*a	   @�o&@   @�o&@      �?!   @�o&@) !=�u_@2sQ��"�%@e��:�(@�������:              �?        
s
generator_loss*a	    7$�    7$�      �?!    7$�) �+Y@2sQ��"�%��ՠ�M�#��������:              �?        ��g8�       b�D�	�(�����A�*�
w
discriminator_loss*a	   @�@   @�@      �?!   @�@)���GI-@2u�rʭ�@�DK��@�������:              �?        
s
generator_loss*a	    ��    ��      �?!    ��) ����(@2�DK���u�rʭ���������:              �?        ���       b�D�	c;�����A�*�
w
discriminator_loss*a	   @9Ƽ?   @9Ƽ?      �?!   @9Ƽ?)�l��߉?2%g�cE9�?��(!�ؼ?�������:              �?        
s
generator_loss*a	   @���   @���      �?!   @���) P�"�=2��~��¾�[�=�k���������:              �?        �9���       �E��	8s�����A�*�
w
discriminator_loss*a	    (��?    (��?      �?!    (��?)  2�uc?2�g���w�?���g��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        G�6H�       �E��	%�ǔ���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) ":���F?2}Y�4j�?��<�A��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ׌<�       �E��	u�Δ���A�*�
w
discriminator_loss*a	   �i�?   �i�?      �?!   �i�?) �W���1?2�#�h/�?���&�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	]�����A�*�
w
discriminator_loss*a	   ��ݑ?   ��ݑ?      �?!   ��ݑ?)@�I���3?2���&�?�Rc�ݒ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        o���       �E��	,�����A�*�
w
discriminator_loss*a	    bh�?    bh�?      �?!    bh�?)@��	��?2���J�\�?-Ա�L�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���>�       �E��	�6����A�*�
w
discriminator_loss*a	   �G�?   �G�?      �?!   �G�?) �	�BO?2���J�\�?-Ա�L�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        x����       �E��	�[����A�*�
w
discriminator_loss*a	   ��*|?   ��*|?      �?!   ��*|?)�L�i[�?2o��5sz?���T}?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        h)�I�       �E��	�z%����A�*�
w
discriminator_loss*a	   `Dۏ?   `Dۏ?      �?!   `Dۏ?) t겶/?2�#�h/�?���&�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        
�E��       �E��		o,����A�*�
w
discriminator_loss*a	   `-lq?   `-lq?      �?!   `-lq?)@�ѻ��>2;8�clp?uWy��r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��ݠ�       �E��	7�3����A�*�
w
discriminator_loss*a	    N;x?    N;x?      �?!    N;x?)  ^�bY?2*QH�x?o��5sz?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �e]�       �E��	t�:����A�*�
w
discriminator_loss*a	   @Hn�?   @Hn�?      �?!   @Hn�?) A����?2>	� �?����=��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        *�BN�       �E��	H�z����A�*�
w
discriminator_loss*a	   ��v?   ��v?      �?!   ��v?)  � ?2&b՞
�u?*QH�x?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        )EѶ�       �E��	�ł����A�*�
w
discriminator_loss*a	    �m?    �m?      �?!    �m?)  �- ��>2ߤ�(g%k?�N�W�m?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �G��       �E��	~r�����A�*�
w
discriminator_loss*a	   `�=v?   `�=v?      �?!   `�=v?)@�����>2&b՞
�u?*QH�x?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��h��       �E��	�{�����A�*�
w
discriminator_loss*a	   �;�o?   �;�o?      �?!   �;�o?) ���z��>2�N�W�m?;8�clp?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �(/��       �E��	�{�����A�*�
w
discriminator_loss*a	   `��m?   `��m?      �?!   `��m?) ��+t��>2ߤ�(g%k?�N�W�m?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Sp��       �E��	ld�����A�*�
w
discriminator_loss*a	   ��^}?   ��^}?      �?!   ��^}?)��|�
?2���T}?>	� �?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �+B��       �E��	�m�����A�*�
w
discriminator_loss*a	   �)q?   �)q?      �?!   �)q?) ����4�>2;8�clp?uWy��r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        2�_��       �E��	c�����A�*�
w
discriminator_loss*a	   `q?   `q?      �?!   `q?) ���%?2���T}?>	� �?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        <)%=�       �E��	7����A�*�
w
discriminator_loss*a	    �as?    �as?      �?!    �as?)@P �z�>2uWy��r?hyO�s?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        5Vb��       �E��	nZ�����A�*�
w
discriminator_loss*a	   ��a?   ��a?      �?!   ��a?) $���3�>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �'��       �E��	&����A�*�
w
discriminator_loss*a	   ��	k?   ��	k?      �?!   ��	k?) R
�K��>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �/$r�       �E��	÷����A�*�
w
discriminator_loss*a	   @Emf?   @Emf?      �?!   @Emf?) ���ho�>25Ucv0ed?Tw��Nof?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        B���       �E��	K�����A�*�
w
discriminator_loss*a	   ��uq?   ��uq?      �?!   ��uq?)@F?,3�>2;8�clp?uWy��r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Ua�Y�       �E��	������A�*�
w
discriminator_loss*a	   ���p?   ���p?      �?!   ���p?) �$���>2;8�clp?uWy��r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �B���       �E��	x�����A�*�
w
discriminator_loss*a	    ��q?    ��q?      �?!    ��q?) y���>2;8�clp?uWy��r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ֓	�       �E��	˵$����A�*�
w
discriminator_loss*a	   �BsY?   �BsY?      �?!   �BsY?)�<Ot�=�>2��bB�SY?�m9�H�[?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7Xu��       �E��	��p����A�*�
w
discriminator_loss*a	   �L�w?   �L�w?      �?!   �L�w?) �F�s?2&b՞
�u?*QH�x?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        y���       �E��	�x����A�*�
w
discriminator_loss*a	   @<ji?   @<ji?      �?!   @<ji?)�p��^/�>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        'i-D�       �E��		�����A�*�
w
discriminator_loss*a	   ��!r?   ��!r?      �?!   ��!r?) ��K��>2uWy��r?hyO�s?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7L�       �E��	�������A�*�
w
discriminator_loss*a	   �h�u?   �h�u?      �?!   �h�u?) ���wC�>2hyO�s?&b՞
�u?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �i�b�       �E��	t������A�*�
w
discriminator_loss*a	   �%lV?   �%lV?      �?!   �%lV?) ��Bl�>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #���       �E��	mw�����A�*�
w
discriminator_loss*a	    �c?    �c?      �?!    �c?) � ݣ�>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	������A�*�
w
discriminator_loss*a	   `vk?   `vk?      �?!   `vk?) Â���>2ߤ�(g%k?�N�W�m?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �yi��       �E��	捣����A�*�
w
discriminator_loss*a	   �g�g?   �g�g?      �?!   �g�g?) �a���>2Tw��Nof?P}���h?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        k�~z�       �E��	�G�����A�*�
w
discriminator_loss*a	   �;�r?   �;�r?      �?!   �;�r?)@2J{�/�>2uWy��r?hyO�s?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	������A�*�
w
discriminator_loss*a	   @٤f?   @٤f?      �?!   @٤f?)�����>2Tw��Nof?P}���h?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        o�c��       �E��	� ����A�*�
w
discriminator_loss*a	   ���h?   ���h?      �?!   ���h?) �!1Z�>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��lv�       �E��	�.����A�*�
w
discriminator_loss*a	   @�k?   @�k?      �?!   @�k?)�<��o��>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]�}��       �E��	�H����A�*�
w
discriminator_loss*a	   ��b?   ��b?      �?!   ��b?) �O����>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �%��       �E��	B����A�*�
w
discriminator_loss*a	   @;�h?   @;�h?      �?!   @;�h?)��a�0?�>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �V:��       �E��	�."����A�*�
w
discriminator_loss*a	   ��uZ?   ��uZ?      �?!   ��uZ?) R�1���>2��bB�SY?�m9�H�[?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �N���       �E��	�&)����A�*�
w
discriminator_loss*a	   �f/f?   �f/f?      �?!   �f/f?) �;����>25Ucv0ed?Tw��Nof?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �f��       �E��	�����A�*�
w
discriminator_loss*a	   ��c?   ��c?      �?!   ��c?)@6�/]�>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �'��       �E��	r������A�*�
w
discriminator_loss*a	   ��=c?   ��=c?      �?!   ��=c?) $��E#�>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        `���       �E��	pŒ����A�*�
w
discriminator_loss*a	   �Bri?   �Bri?      �?!   �Bri?)�<#H <�>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Xז��       �E��	�Ǚ����A�*�
w
discriminator_loss*a	    �z[?    �z[?      �?!    �z[?) ʞŘ�>2��bB�SY?�m9�H�[?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �lp��       �E��	Sՠ����A�*�
w
discriminator_loss*a	   @�]X?   @�]X?      �?!   @�]X?)���֍�>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        @Ӵ|�       �E��	�ڧ����A�*�
w
discriminator_loss*a	   �-Va?   �-Va?      �?!   �-Va?) d�����>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �
��       �E��	�������A�*�
w
discriminator_loss*a	   �.�b?   �.�b?      �?!   �.�b?) ���m��>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �4!��       �E��	@������A�*�
w
discriminator_loss*a	   ��-^?   ��-^?      �?!   ��-^?)���v�>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �	��       �E��	�O����A�*�
w
discriminator_loss*a	   @�Da?   @�Da?      �?!   @�Da?) y��~��>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �~�p�       �E��	����A�*�
w
discriminator_loss*a	    ��[?    ��[?      �?!    ��[?) ��Q���>2��bB�SY?�m9�H�[?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��i��       �E��	j$����A�*�
w
discriminator_loss*a	   ��cj?   ��cj?      �?!   ��cj?)��D���>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        r�}2�       �E��	�j+����A�*�
w
discriminator_loss*a	   @Ǟ^?   @Ǟ^?      �?!   @Ǟ^?)��pl�L�>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �<G�       �E��	]e2����A�*�
w
discriminator_loss*a	   ���\?   ���\?      �?!   ���\?) �c��G�>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �KҾ�       �E��	.F9����A�*�
w
discriminator_loss*a	   @.d?   @.d?      �?!   @.d?) ���|�>2���%��b?5Ucv0ed?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�I@����A�*�
w
discriminator_loss*a	   @AU?   @AU?      �?!   @AU?) ����;�>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        F߫s�       �E��	��G����A�*�
w
discriminator_loss*a	    �VV?    �VV?      �?!    �VV?) ����0�>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        c��^�       �E��	$�����A�*�
w
discriminator_loss*a	   ��>Y?   ��>Y?      �?!   ��>Y?) �Ś��>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        O-Q��       �E��	�׵����A�*�
w
discriminator_loss*a	   @!TG?   @!TG?      �?!   @!TG?)������>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	������A�*�
w
discriminator_loss*a	   �|�W?   �|�W?      �?!   �|�W?)�TFm��>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��t��       �E��	6Ę���A�*�
w
discriminator_loss*a	   @��V?   @��V?      �?!   @��V?)�����0�>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�0˘���A�*�
w
discriminator_loss*a	    \T?    \T?      �?!    \T?) �C<�>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        	�Gy�       �E��	*4Ҙ���A�*�
w
discriminator_loss*a	   �R�Z?   �R�Z?      �?!   �R�Z?) �BE��>2��bB�SY?�m9�H�[?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �d���       �E��	�h٘���A�*�
w
discriminator_loss*a	   @��Q?   @��Q?      �?!   @��Q?) �᧿�>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �)���       �E��	S������A�*�
w
discriminator_loss*a	    k�G?    k�G?      �?!    k�G?) ȕ�!��>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #$��       �E��	�O����A�*�
w
discriminator_loss*a	   @}zg?   @}zg?      �?!   @}zg?)�<���9�>2Tw��Nof?P}���h?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        f͜x�       �E��	[V����A�*�
w
discriminator_loss*a	    \Q?    \Q?      �?!    \Q?)@pJ��.�>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        z���       �E��	�]����A�*�
w
discriminator_loss*a	    �_?    �_?      �?!    �_?)  >l��>2E��{��^?�l�P�`?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        A8���       �E��	�d����A�*�
w
discriminator_loss*a	   �>�`?   �>�`?      �?!   �>�`?) <�z�>2E��{��^?�l�P�`?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        X���       �E��	��k����A�*�
w
discriminator_loss*a	   ��i?   ��i?      �?!   ��i?)�4��&��>2P}���h?ߤ�(g%k?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �$|��       �E��	Ǟr����A�*�
w
discriminator_loss*a	   @z�D?   @z�D?      �?!   @z�D?) ��gt�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        h���       �E��	H�y����A�*�
w
discriminator_loss*a	   `d{U?   `d{U?      �?!   `d{U?)@���׼>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��yk�       �E��	:耙���A�*�
w
discriminator_loss*a	    "�J?    "�J?      �?!    "�J?) d�9.y�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	GD�����A�*�
w
discriminator_loss*a	    a?    a?      �?!    a?)@�j���>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �!�       �E��	b������A�*�
w
discriminator_loss*a	    ��`?    ��`?      �?!    ��`?)  @����>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � ���       �E��	�����A�*�
w
discriminator_loss*a	   @�\?   @�\?      �?!   @�\?)��>�p��>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	������A�*�
w
discriminator_loss*a	    ��H?    ��H?      �?!    ��H?)  �e{"�>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �6:�       �E��	3����A�*�
w
discriminator_loss*a	    ��L?    ��L?      �?!    ��L?) ���,�>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ߈��       �E��	{W����A�*�
w
discriminator_loss*a	   �
]L?   �
]L?      �?!   �
]L?) �9��#�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �'ur�       �E��	�I!����A�*�
w
discriminator_loss*a	   ���N?   ���N?      �?!   ���N?) V(�.R�>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?         <n�       �E��	�B(����A�*�
w
discriminator_loss*a	   ��7Y?   ��7Y?      �?!   ��7Y?) "�o���>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        T�X�       �E��	S������A�*�
w
discriminator_loss*a	   `�H?   `�H?      �?!   `�H?) �@L4�>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �aݜ�       �E��	(�����A�*�
w
discriminator_loss*a	    �;?    �;?      �?!    �;?) ,<��>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        !m��       �E��	�Q�����A�*�
w
discriminator_loss*a	    ��Q?    ��Q?      �?!    ��Q?) �]q�޳>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �l��       �E��	�������A�*�
w
discriminator_loss*a	   `ӿ]?   `ӿ]?      �?!   `ӿ]?) ;�-��>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        P?{��       �E��	ą�����A�*�
w
discriminator_loss*a	   ��Q?   ��Q?      �?!   ��Q?) ċY��>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        d�~�       �E��	��Ś���A�*�
w
discriminator_loss*a	    2F?    2F?      �?!    2F?) @��,ʞ>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Xm���       �E��	Y�̚���A�*�
w
discriminator_loss*a	   @D�Q?   @D�Q?      �?!   @D�Q?) !+b׳>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��9V�       �E��	�Ӛ���A�*�
w
discriminator_loss*a	   �bM?   �bM?      �?!   �bM?) 2?I�w�>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��^�       �E��	��]����A�*�
w
discriminator_loss*a	   @��R?   @��R?      �?!   @��R?) ���>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �|�       �E��	�d����A�*�
w
discriminator_loss*a	    ��X?    ��X?      �?!    ��X?) a{� �>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        4���       �E��	�Dl����A�*�
w
discriminator_loss*a	    �\?    �\?      �?!    �\?) ����>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �
�       �E��	�ps����A�*�
w
discriminator_loss*a	   `q\E?   `q\E?      �?!   `q\E?)@^󶿄�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��#�       �E��	�j|����A�*�
w
discriminator_loss*a	   @wb?   @wb?      �?!   @wb?) ɨRƄ�>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	3{�����A�*�
w
discriminator_loss*a	   ���T?   ���T?      �?!   ���T?)@8�L�غ>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��k��       �E��	�������A�*�
w
discriminator_loss*a	    ӃL?    ӃL?      �?!    ӃL?) H�L�h�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �H�       �E��	tQ�����A�*�
w
discriminator_loss*a	   `��<?   `��<?      �?!   `��<?) ��30�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        nj1��       �E��	'�;����A�*�
w
discriminator_loss*a	    ]K;?    ]K;?      �?!    ]K;?) ���G�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	F�C����A�*�
w
discriminator_loss*a	    ��D?    ��D?      �?!    ��D?) @N0��>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �s`��       �E��	FXK����A�*�
w
discriminator_loss*a	   `��S?   `��S?      �?!   `��S?)@�Y����>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �.}��       �E��	�FS����A�*�
w
discriminator_loss*a	   ��/2?   ��/2?      �?!   ��/2?) ���t>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        >�G�       �E��	vpZ����A�*�
w
discriminator_loss*a	   ��VG?   ��VG?      �?!   ��VG?)�\��h�>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �E��       �E��	�Bb����A�*�
w
discriminator_loss*a	   �iN?   �iN?      �?!   �iN?) �Y��>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��r(�       �E��	��j����A�*�
w
discriminator_loss*a	   ��3D?   ��3D?      �?!   ��3D?)@�y��>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        `��r�       �E��	Hws����A�*�
w
discriminator_loss*a	   �iX?   �iX?      �?!   �iX?) �~��>2ܗ�SsW?��bB�SY?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��_��       �E��	0;����A�*�
w
discriminator_loss*a	   �t~\?   �t~\?      �?!   �t~\?) �B@_�>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	Ǟ����A�*�
w
discriminator_loss*a	   �%�S?   �%�S?      �?!   �%�S?) ���>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Lt�       �E��	W"����A�*�
w
discriminator_loss*a	   `%J>?   `%J>?      �?!   `%J>?) �ᱫ�>2d�\D�X=?���#@?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �[q7�       �E��	��%����A�*�
w
discriminator_loss*a	   ��T?   ��T?      �?!   ��T?)@DK��>2�lDZrS?<DKc��T?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        K^��       �E��	��,����A�*�
w
discriminator_loss*a	   �a�B?   �a�B?      �?!   �a�B?)@�&뫀�>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        i���       �E��	%4����A�*�
w
discriminator_loss*a	   @�WD?   @�WD?      �?!   @�WD?) y��ݙ>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �)��       �E��	�(;����A�*�
w
discriminator_loss*a	   �)L?   �)L?      �?!   �)L?)�x�iئ�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	LpB����A�*�
w
discriminator_loss*a	   ��G?   ��G?      �?!   ��G?) �6�߿�>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        0���       �E��	|�읨��A�*�
w
discriminator_loss*a	   @�G?   @�G?      �?!   @�G?)�4��N��>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	h������A�*�
w
discriminator_loss*a	   @\Q?   @\Q?      �?!   @\Q?) ��.�>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        e_���       �E��	�-�����A�*�
w
discriminator_loss*a	   ���;?   ���;?      �?!   ���;?) >keiW�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��"J�       �E��	�D����A�*�
w
discriminator_loss*a	    ~�E?    ~�E?      �?!    ~�E?) @`���>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �BIF�       �E��	؅����A�*�
w
discriminator_loss*a	   �2HM?   �2HM?      �?!   �2HM?)�|��~˪>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �.c�       �E��	�����A�*�
w
discriminator_loss*a	   @u�V?   @u�V?      �?!   @u�V?)��0��>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �?��       �E��	&&����A�*�
w
discriminator_loss*a	   `\?   `\?      �?!   `\?) �34؈�>2�m9�H�[?E��{��^?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �w��       �E��	w(����A�*�
w
discriminator_loss*a	   ��D?   ��D?      �?!   ��D?)@\L�E�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �K�R�       �E��	YϞ���A�*�
w
discriminator_loss*a	   `5�@?   `5�@?      �?!   `5�@?)@�i7��>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �2x��       �E��	��מ���A�*�
w
discriminator_loss*a	    �$@?    �$@?      �?!    �$@?)  ��J�>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        <K�       �E��	�=ឨ��A�*�
w
discriminator_loss*a	    ]a?    ]a?      �?!    ]a?) �|��A�>2�l�P�`?���%��b?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �uc	�       �E��	��鞨��A�*�
w
discriminator_loss*a	   �g�J?   �g�J?      �?!   �g�J?) ¦$0�>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �C��       �E��	�����A�*�
w
discriminator_loss*a	   @%�V?   @%�V?      �?!   @%�V?)�\����>2<DKc��T?ܗ�SsW?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �9��       �E��	R������A�*�
w
discriminator_loss*a	   @�+?   @�+?      �?!   @�+?)��_��f>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �M̍�       �E��	������A�*�
w
discriminator_loss*a	   @�D?   @�D?      �?!   @�D?) 9�K[�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7����       �E��	������A�*�
w
discriminator_loss*a	   �W+<?   �W+<?      �?!   �W+<?) �m̈>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        8٣F�       �E��	D����A�*�
w
discriminator_loss*a	    ��H?    ��H?      �?!    ��H?) H����>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ym�`�       �E��	9x˟���A�*�
w
discriminator_loss*a	   ��SF?   ��SF?      �?!   ��SF?)@R%0�'�>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Ço�       �E��	�1ӟ���A�*�
w
discriminator_loss*a	   �tC!?   �tC!?      �?!   �tC!?) D �s�R>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        a��       M�	�U韨��A*�
w
discriminator_loss*a	    ��8?    ��8?      �?!    ��8?) ��f{�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �V�       ���g	����A*�
w
discriminator_loss*a	    �K0?    �K0?      �?!    �K0?) �6�p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        WX��       ���g	x�����A
*�
w
discriminator_loss*a	   ���P?   ���P?      �?!   ���P?) Y��m��>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �h�
�       ���g	ʂ�����A*�
w
discriminator_loss*a	   @��R?   @��R?      �?!   @��R?) �%{/�>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��E��       ���g	!�����A*�
w
discriminator_loss*a	    ��I?    ��I?      �?!    ��I?)  �g8�>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ,:!�       ���g	�4�����A*�
w
discriminator_loss*a	    �zP?    �zP?      �?!    �zP?)  ��w��>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��=�       ���g	�l�����A*�
w
discriminator_loss*a	   ���I?   ���I?      �?!   ���I?)�D+�;��>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        =1�l�       ���g	�7à���A#*�
w
discriminator_loss*a	   ���8?   ���8?      �?!   ���8?) 2_��q�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �d#�       ���g	?�ʠ���A(*�
w
discriminator_loss*a	   ���8?   ���8?      �?!   ���8?) �O8k�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��'��       ���g	�ْ����A-*�
w
discriminator_loss*a	   @YS.?   @YS.?      �?!   @YS.?)���o �l>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �h���       ���g	p~�����A2*�
w
discriminator_loss*a	   ��9?   ��9?      �?!   ��9?) (�
��>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��4!�       ���g	������A7*�
w
discriminator_loss*a	    �??    �??      �?!    �??)  ��p��>2d�\D�X=?���#@?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �c���       ���g	Q�����A<*�
w
discriminator_loss*a	   ���R?   ���R?      �?!   ���R?) Ķ���>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Oap]�       ���g	첰����AA*�
w
discriminator_loss*a	    �4?    �4?      �?!    �4?) @�u3"{>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        j-�(�       ���g	S������AF*�
w
discriminator_loss*a	    �$@?    �$@?      �?!    �$@?)@���I�>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �q��       ���g	s�����AK*�
w
discriminator_loss*a	   @��2?   @��2?      �?!   @��2?) �.��v>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��?��       ���g	�~ǡ���AP*�
w
discriminator_loss*a	   @<;?   @<;?      �?!   @<;?)�p�;&φ>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        O�s�       ���g	�������AU*�
w
discriminator_loss*a	   ���L?   ���L?      �?!   ���L?)�(�{��>2IcD���L?k�1^�sO?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        /2��       ���g	������AZ*�
w
discriminator_loss*a	    �_R?    �_R?      �?!    �_R?) �:�7�>2nK���LQ?�lDZrS?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��!Z�       ���g	c������A_*�
w
discriminator_loss*a	    ~�@?    ~�@?      �?!    ~�@?)@8�ר��>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       ���g	Y������Ad*�
w
discriminator_loss*a	   ��T6?   ��T6?      �?!   ��T6?) D�%�*>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        BJu��       ���g	#������Ai*�
w
discriminator_loss*a	   @N�9?   @N�9?      �?!   @N�9?)�X���ׄ>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        v�y��       ���g	�������An*�
w
discriminator_loss*a	   ��C?   ��C?      �?!   ��C?) IZc��>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Q~�h�       ���g	�������As*�
w
discriminator_loss*a	   �z�A?   �z�A?      �?!   �z�A?)@ίxh��>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �`G�       ���g	������Ax*�
w
discriminator_loss*a	   ��+?   ��+?      �?!   ��+?)�\����g>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        P���       ���g	�5w����A}*�
w
discriminator_loss*a	   ��YB?   ��YB?      �?!   ��YB?) q��k�>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ?G��       �E��	�T����A�*�
w
discriminator_loss*a	    �EC?    �EC?      �?!    �EC?) @��U6�>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �p��       �E��	il�����A�*�
w
discriminator_loss*a	   ��n5?   ��n5?      �?!   ��n5?) Č8��|>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �I���       �E��	�������A�*�
w
discriminator_loss*a	    [�9?    [�9?      �?!    [�9?) ���〄>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        *.6�       �E��	�q�����A�*�
w
discriminator_loss*a	   ���A?   ���A?      �?!   ���A?) Q:z��>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �@��       �E��	�$�����A�*�
w
discriminator_loss*a	    �1?    �1?      �?!    �1?)@�wۥFr>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �x���       �E��	�������A�*�
w
discriminator_loss*a	   ��-?   ��-?      �?!   ��-?) ���mj>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �\]'�       �E��	ٽ�����A�*�
w
discriminator_loss*a	   �#8P?   �#8P?      �?!   �#8P?) ���q�>2k�1^�sO?nK���LQ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Jĵ��       �E��	f-w����A�*�
w
discriminator_loss*a	   @I]7?   @I]7?      �?!   @I]7?)��k>)�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��9 �       �E��	��~����A�*�
w
discriminator_loss*a	    �!F?    �!F?      �?!    �!F?) ��a��>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        L�0�       �E��	�L�����A�*�
w
discriminator_loss*a	   �p0B?   �p0B?      �?!   �p0B?) ��W���>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��4��       �E��	XA�����A�*�
w
discriminator_loss*a	   �!�:?   �!�:?      �?!   �!�:?)��S5~�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �o���       �E��	*�����A�*�
w
discriminator_loss*a	    zf2?    zf2?      �?!    zf2?) @"�")u>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Cg��       �E��	^�����A�*�
w
discriminator_loss*a	   �u3?   �u3?      �?!   �u3?) �����v>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        l��9�       �E��	�t�����A�*�
w
discriminator_loss*a	    ��C?    ��C?      �?!    ��C?)@��Δ�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        e���       �E��	�����A�*�
w
discriminator_loss*a	   @X"?   @X"?      �?!   @X"?) A���U>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �5ǥ�       �E��	䈥���A�*�
w
discriminator_loss*a	   ��{4?   ��{4?      �?!   ��{4?)@�ߣ9z>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �"�       �E��	�𑥨��A�*�
w
discriminator_loss*a	   ���(?   ���(?      �?!   ���(?) 2�b�b>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��s�       �E��	������A�*�
w
discriminator_loss*a	    �F8?    �F8?      �?!    �F8?)  n��j�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	o)�����A�*�
w
discriminator_loss*a	   �rF?   �rF?      �?!   �rF?) ��@W�>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��G��       �E��	�,�����A�*�
w
discriminator_loss*a	   �(�D?   �(�D?      �?!   �(�D?) ���F�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ܎�Y�       �E��	`&�����A�*�
w
discriminator_loss*a	   ��F?   ��F?      �?!   ��F?) �bQ�q�>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]9{�       �E��	������A�*�
w
discriminator_loss*a	   @i	F?   @i	F?      �?!   @i	F?) Y��Y�>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        q�{K�       �E��	�-�����A�*�
w
discriminator_loss*a	   �7"K?   �7"K?      �?!   �7"K?)� ٪��>2�qU���I?IcD���L?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��]��       �E��	Sp�����A�*�
w
discriminator_loss*a	   ��O)?   ��O)?      �?!   ��O)?) ���d>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?         ���       �E��	�`�����A�*�
w
discriminator_loss*a	   @�B?   @�B?      �?!   @�B?) i�=��>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ֗���       �E��	A������A�*�
w
discriminator_loss*a	   ��A0?   ��A0?      �?!   ��A0?) ��+�p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �`���       �E��	~������A�*�
w
discriminator_loss*a	   �E 0?   �E 0?      �?!   �E 0?)@�.A� p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��X�       �E��	 ������A�*�
w
discriminator_loss*a	   �=�:?   �=�:?      �?!   �=�:?)�(C��/�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        !Q���       �E��	:������A�*�
w
discriminator_loss*a	   @\�@?   @\�@?      �?!   @\�@?) �#���>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �U���       �E��	�������A�*�
w
discriminator_loss*a	   �B�1?   �B�1?      �?!   �B�1?)@n�N#�s>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �u��       �E��	�Ȧ���A�*�
w
discriminator_loss*a	   ���H?   ���H?      �?!   ���H?) b���ڢ>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	x춧���A�*�
w
discriminator_loss*a	   ��4C?   ��4C?      �?!   ��4C?)@6���>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �/��       �E��	�������A�*�
w
discriminator_loss*a	   �}7?   �}7?      �?!   �}7?)�(�֩�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        C�h��       �E��	��Ƨ���A�*�
w
discriminator_loss*a	   ���<?   ���<?      �?!   ���<?)�lk_Hŉ>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        5WJ�       �E��	Ч���A�*�
w
discriminator_loss*a	   @E�E?   @E�E?      �?!   @E�E?) �+G���>2a�$��{E?
����G?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �չ��       �E��	)#ا���A�*�
w
discriminator_loss*a	   �@�<?   �@�<?      �?!   �@�<?) ��F��>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        /���       �E��	�c᧨��A�*�
w
discriminator_loss*a	   ��<?   ��<?      �?!   ��<?) �ϲ-��>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � ��       �E��	�u駨��A�*�
w
discriminator_loss*a	   ���!?   ���!?      �?!   ���!?)@��DS>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �2��       �E��	O���A�*�
w
discriminator_loss*a	    �#?    �#?      �?!    �#?) �
P��W>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        9E} �       �E��	�����A�*�
w
discriminator_loss*a	   ��3?   ��3?      �?!   ��3?) A�oYx>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�3����A�*�
w
discriminator_loss*a	   �Ul#?   �Ul#?      �?!   �Ul#?) �k/)�W>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        sM�       �E��	����A�*�
w
discriminator_loss*a	   ��!?   ��!?      �?!   ��!?)@�-6�@R>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Bn��       �E��	b����A�*�
w
discriminator_loss*a	   �5�?   �5�?      �?!   �5�?) ryr�%H>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �G���       �E��	O/$����A�*�
w
discriminator_loss*a	   `�G1?   `�G1?      �?!   `�G1?)@ZM⾩r>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        &�$��       �E��	5+����A�*�
w
discriminator_loss*a	    �@?    �@?      �?!    �@?) @j���>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        LCkm�       �E��	�2����A�*�
w
discriminator_loss*a	   �߫"?   �߫"?      �?!   �߫"?)@����U>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�9����A�*�
w
discriminator_loss*a	   �6?   �6?      �?!   �6?)@��TJw~>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        rY� �       �E��	��+����A�*�
w
discriminator_loss*a	   `ľ+?   `ľ+?      �?!   `ľ+?) ��\h>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        E+y�       �E��	Ck3����A�*�
w
discriminator_loss*a	   ��V5?   ��V5?      �?!   ��V5?) y���t|>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        "գ�       �E��	�:����A�*�
w
discriminator_loss*a	   @e�.?   @e�.?      �?!   @e�.?)�\|�7m>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�"B����A�*�
w
discriminator_loss*a	    ^K0?    ^K0?      �?!    ^K0?) @h�p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        y���       �E��	>6I����A�*�
w
discriminator_loss*a	   ��4#?   ��4#?      �?!   ��4#?) ���W>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        (��X�       �E��	�TP����A�*�
w
discriminator_loss*a	   @R�+?   @R�+?      �?!   @R�+?)�hc<�h>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        T��y�       �E��	�[W����A�*�
w
discriminator_loss*a	   �-UC?   �-UC?      �?!   �-UC?) њ\�>2�!�A?�T���C?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �=���       �E��	Uf^����A�*�
w
discriminator_loss*a	    �Q9?    �Q9?      �?!    �Q9?) �m��>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        pC���       �E��	o8i����A�*�
w
discriminator_loss*a	   �� %?   �� %?      �?!   �� %?)@Ȅ��[>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ԓ�h�       �E��	��q����A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) ��V�1>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        H�G�       �E��	Ɔz����A�*�
w
discriminator_loss*a	   � �"?   � �"?      �?!   � �"?) 	�V>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        0!g0�       �E��	E�����A�*�
w
discriminator_loss*a	   ��Q/?   ��Q/?      �?!   ��Q/?)��D|��n>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        K$5�       �E��	1n�����A�*�
w
discriminator_loss*a	    �-?    �-?      �?!    �-?) 
���pj>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��~�       �E��	�r�����A�*�
w
discriminator_loss*a	   ��X)?   ��X)?      �?!   ��X)?) �d>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        :䘝�       �E��	>j�����A�*�
w
discriminator_loss*a	   �e�I?   �e�I?      �?!   �e�I?)��K��h�>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        wť��       �E��	�d�����A�*�
w
discriminator_loss*a	    �b1?    �b1?      �?!    �b1?) �c��r>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���9�       �E��	�믬���A�*�
w
discriminator_loss*a	   ���3?   ���3?      �?!   ���3?)@�w��w>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        3����       �E��	P�����A�*�
w
discriminator_loss*a	   �y�D?   �y�D?      �?!   �y�D?)@��f�>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        6`�\�       �E��	ٻ�����A�*�
w
discriminator_loss*a	    �u9?    �u9?      �?!    �u9?) ���B�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	��Ƭ���A�*�
w
discriminator_loss*a	   �Z_$?   �Z_$?      �?!   �Z_$?) 䯄��Y>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        p��       �E��	��ͬ���A�*�
w
discriminator_loss*a	   ���5?   ���5?      �?!   ���5?) �I�}>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]'m"�       �E��	�Ԭ���A�*�
w
discriminator_loss*a	    ��0?    ��0?      �?!    ��0?)@�ǘ�q>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Hh�5�       �E��	�ܬ���A�*�
w
discriminator_loss*a	    �;?    �;?      �?!    �;?)  Ju�=�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        htږ�       �E��	2�⬨��A�*�
w
discriminator_loss*a	   � �6?   � �6?      �?!   � �6?) C)ˢ�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        J��       �E��	������A�*�
w
discriminator_loss*a	   ��� ?   ��� ?      �?!   ��� ?) a̡Q>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �(k�       �E��	�	����A�*�
w
discriminator_loss*a	   @�Z0?   @�Z0?      �?!   @�Z0?) Y7T�p>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Q?.�       �E��	~N����A�*�
w
discriminator_loss*a	   @�1?   @�1?      �?!   @�1?) !eN�r>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        wb'��       �E��	`V����A�*�
w
discriminator_loss*a	   ��_?   ��_?      �?!   ��_?) �Ž��9>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��P��       �E��	�k����A�*�
w
discriminator_loss*a	    V�1?    V�1?      �?!    V�1?) @��
t>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���V�       �E��	{p&����A�*�
w
discriminator_loss*a	    n ?    n ?      �?!    n ?) |�3�P>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �N�       �E��	qn-����A�*�
w
discriminator_loss*a	   �l.+?   �l.+?      �?!   �l.+?) nv�g>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��s�       �E��	ud4����A�*�
w
discriminator_loss*a	   `Ǳ%?   `Ǳ%?      �?!   `Ǳ%?)@f�bj]>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��y�       �E��	�D@����A�*�
w
discriminator_loss*a	   @�8?   @�8?      �?!   @�8?)�Lo^�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	P�H����A�*�
w
discriminator_loss*a	   ���+?   ���+?      �?!   ���+?)��P�P�g>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        <�&��       �E��	P�R����A�*�
w
discriminator_loss*a	    $�0?    $�0?      �?!    $�0?)  Qr��q>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ç(U�       �E��	�Z����A�*�
w
discriminator_loss*a	    ��#?    ��#?      �?!    ��#?)@�t��xX>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ɼ���       �E��	��c����A�*�
w
discriminator_loss*a	   @�Z%?   @�Z%?      �?!   @�Z%?) AX�\>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �O���       �E��	�l����A�*�
w
discriminator_loss*a	   �?   �?      �?!   �?) ��T�j4>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ժ��       �E��	Rv����A�*�
w
discriminator_loss*a	    �2?    �2?      �?!    �2?) @^�Тu>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �k���       �E��	ؚ~����A�*�
w
discriminator_loss*a	   �(&?   �(&?      �?!   �(&?) Ā�m�^>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        D|dJ�       �E��	�������A�*�
w
discriminator_loss*a	   ��+?   ��+?      �?!   ��+?) �<�1h>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	'g�����A�*�
w
discriminator_loss*a	   @�@7?   @�@7?      �?!   @�@7?)�h��T�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��{�       �E��	��ǰ���A�*�
w
discriminator_loss*a	   ��F/?   ��F/?      �?!   ��F/?)����W�n>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        `���       �E��	�а���A�*�
w
discriminator_loss*a	   ��2?   ��2?      �?!   ��2?)@��bnt>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�]װ���A�*�
w
discriminator_loss*a	    A7?    A7?      �?!    A7?) �婢>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %��       �E��	^�ް���A�*�
w
discriminator_loss*a	   �be?   �be?      �?!   �be?) d��&5>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        5�X�       �E��	x氨��A�*�
w
discriminator_loss*a	   @��(?   @��(?      �?!   @��(?)�P�xWLc>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        C]�p�       �E��	M������A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �HjF|M>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        6NV�       �E��	B-����A�*�
w
discriminator_loss*a	   �ń0?   �ń0?      �?!   �ń0?)@(��q>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        	t���       �E��	_����A�*�
w
discriminator_loss*a	   @zQ+?   @zQ+?      �?!   @zQ+?)���MRg>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        fy[��       �E��	=f$����A�*�
w
discriminator_loss*a	   @c#"?   @c#"?      �?!   @c#"?) ����T>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        >��       �E��	��,����A�*�
w
discriminator_loss*a	   �V';?   �V';?      �?!   �V';?) ڭ��
�>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Uy�       �E��	JX5����A�*�
w
discriminator_loss*a	   ���)?   ���)?      �?!   ���)?) 2/���d>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]h���       �E��	�b<����A�*�
w
discriminator_loss*a	   �}�?   �}�?      �?!   �}�?) $���O>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        \�-��       �E��	�AD����A�*�
w
discriminator_loss*a	   �̻D?   �̻D?      �?!   �̻D?) )D,ޚ>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        1��]�       �E��	�)L����A�*�
w
discriminator_loss*a	    P�(?    P�(?      �?!    P�(?) ��+�b>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        L'[�       �E��	h�p����A�*�
w
discriminator_loss*a	   @{�3?   @{�3?      �?!   @{�3?) i�Wz�x>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	C�x����A�*�
w
discriminator_loss*a	   �<.?   �<.?      �?!   �<.?)�T�C�Ml>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %�2�       �E��		�����A�*�
w
discriminator_loss*a	   �,23?   �,23?      �?!   �,23?) )-��w>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        +(:6�       �E��	�������A�*�
w
discriminator_loss*a	   �~Q?   �~Q?      �?!   �~Q?) \=�E>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �A.�       �E��	�'�����A�*�
w
discriminator_loss*a	   �p�?   �p�?      �?!   �p�?) ����B>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        g�a?�       �E��	�^�����A�*�
w
discriminator_loss*a	    ��#?    ��#?      �?!    ��#?)  ���X>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��@�       �E��	Q������A�*�
w
discriminator_loss*a	   ��09?   ��09?      �?!   ��09?) $~��ԃ>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        -�,��       �E��	iX�����A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) �Q��H>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �u5P�       �E��	K�����A�*�
w
discriminator_loss*a	   ���@?   ���@?      �?!   ���@?)@����O�>2���#@?�!�A?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7�S��       �E��	�>�����A�*�
w
discriminator_loss*a	    d'?    d'?      �?!    d'?) �8�ԙ`>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        aV���       �E��	� �����A�*�
w
discriminator_loss*a	   ���-?   ���-?      �?!   ���-?) r�u��k>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ^���       �E��	����A�*�
w
discriminator_loss*a	   `5M!?   `5M!?      �?!   `5M!?)@n���R>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�����A�*�
w
discriminator_loss*a	    wX&?    wX&?      �?!    wX&?)@�ֹ05_>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        X�F[�       �E��	7�����A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)�����M>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �0$�       �E��	e#����A�*�
w
discriminator_loss*a	    wS'?    wS'?      �?!    wS'?) v��� a>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��(�       �E��	l�,����A�*�
w
discriminator_loss*a	   @9L'?   @9L'?      �?!   @9L'?)�lV�G�`>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �*���       �E��	>_����A�*�
w
discriminator_loss*a	   @w�5?   @w�5?      �?!   @w�5?) �X���}>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Ͽ��       �E��	�gg����A�*�
w
discriminator_loss*a	   @��+?   @��+?      �?!   @��+?)���ᶤg>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���q�       �E��	,p����A�*�
w
discriminator_loss*a	   �U*?   �U*?      �?!   �U*?)���N�H>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        I�D�       �E��	jw����A�*�
w
discriminator_loss*a	    [�#?    [�#?      �?!    [�#?) ����kX>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        w�t�       �E��	)�����A�*�
w
discriminator_loss*a	   @�*?   @�*?      �?!   @�*?)��\�rCe>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �AB�       �E��	va�����A�*�
w
discriminator_loss*a	   @�i?   @�i?      �?!   @�i?) I6��e?>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	^M�����A�*�
w
discriminator_loss*a	   ��`?   ��`?      �?!   ��`?)����U�J>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        o�34�       �E��	�9�����A�*�
w
discriminator_loss*a	   @4�?   @4�?      �?!   @4�?)�P���)O>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �c���       �E��	�}ķ���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)����r�A>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ݛ���       �E��	��̷���A�*�
w
discriminator_loss*a	    je(?    je(?      �?!    je(?)  f`�b>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��,}�       �E��	�շ���A�*�
w
discriminator_loss*a	   �@�"?   �@�"?      �?!   �@�"?)@����U>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��:��       �E��	��ܷ���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@XOgx6>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        d���       �E��	*䷨��A�*�
w
discriminator_loss*a	   ��W/?   ��W/?      �?!   ��W/?)���&#�n>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Շ��       �E��	��뷨��A�*�
w
discriminator_loss*a	   `K[?   `K[?      �?!   `K[?) �c�.>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        8�b��       �E��	t7���A�*�
w
discriminator_loss*a	    �~?    �~?      �?!    �~?)  H�PD>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �[�       �E��	RT�����A�*�
w
discriminator_loss*a	   `k?   `k?      �?!   `k?)@z���:>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��E��       �E��	�'����A�*�
w
discriminator_loss*a	   @��1?   @��1?      �?!   @��1?) y�{��s>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        i��       �E��	W�0����A�*�
w
discriminator_loss*a	   �q�?   �q�?      �?!   �q�?) <�>}7H>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %>�*�       �E��	~�9����A�*�
w
discriminator_loss*a	   ��r%?   ��r%?      �?!   ��r%?)@��m�\>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��h��       �E��	�,B����A�*�
w
discriminator_loss*a	    ��*?    ��*?      �?!    ��*?) �WK�f>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �[��       �E��	��J����A�*�
w
discriminator_loss*a	   @�J3?   @�J3?      �?!   @�J3?) Q={�Bw>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �8���       �E��	[6S����A�*�
w
discriminator_loss*a	   �0?   �0?      �?!   �0?)�D��v�@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��<��       �E��	y�[����A�*�
w
discriminator_loss*a	   ��69?   ��69?      �?!   ��69?) ��y�݃>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �DYR�       �E��	��b����A�*�
w
discriminator_loss*a	   �'?   �'?      �?!   �'?) ��jԋ`>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �[���       �E��	��Ǻ���A�*�
w
discriminator_loss*a	   @�!?   @�!?      �?!   @�!?) Q����S>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �,���       �E��	eJк���A�*�
w
discriminator_loss*a	   @E?   @E?      �?!   @E?)�@�d�N>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �@�
�       �E��	�S׺���A�*�
w
discriminator_loss*a	   ���"?   ���"?      �?!   ���"?) d�X�U>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        0���       M�	,���A*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)�Hl��M>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ",��       ���g	��4����A*�
w
discriminator_loss*a	   �\� ?   �\� ?      �?!   �\� ?)@6<tֵ>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �}c�       ���g	a�<����A
*�
w
discriminator_loss*a	   @��%?   @��%?      �?!   @��%?) 	s8&:^>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��C�       ���g	��C����A*�
w
discriminator_loss*a	    ��(?    ��(?      �?!    ��(?) H|��b>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Ʊ�       ���g	��J����A*�
w
discriminator_loss*a	   �b�?   �b�?      �?!   �b�?)���	|A>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        f�)E�       ���g	��Q����A*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) �b�o1>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��W�       ���g	w�X����A*�
w
discriminator_loss*a	    ��,?    ��,?      �?!    ��,?) t���i>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        t/���       ���g	�Z`����A#*�
w
discriminator_loss*a	   @��-?   @��-?      �?!   @��-?)�<�Dl>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �T��       ���g	��g����A(*�
w
discriminator_loss*a	    �v ?    �v ?      �?!    �v ?)  �o;�P>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��|��       ���g	I������A-*�
w
discriminator_loss*a	   ��+!?   ��+!?      �?!   ��+!?) *�lmR>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        R�d�       ���g	�ɽ���A2*�
w
discriminator_loss*a	   @�R
?   @�R
?      �?!   @�R
?)��o��%>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �E�       ���g	<�ѽ���A7*�
w
discriminator_loss*a	   �	2H?   �	2H?      �?!   �	2H?) ��\K�>2
����G?�qU���I?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �#F��       ���g	^�ڽ���A<*�
w
discriminator_loss*a	   ��f?   ��f?      �?!   ��f?) �3��K>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        z���       ���g	�U㽨��AA*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) �38>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]Mw��       ���g	j콨��AF*�
w
discriminator_loss*a	   @TG?   @TG?      �?!   @TG?) �s���9>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��U�       ���g	������AK*�
w
discriminator_loss*a	   ��f?   ��f?      �?!   ��f?) ��wG>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ͺN�       ���g	�o�����AP*�
w
discriminator_loss*a	   �aW/?   �aW/?      �?!   �aW/?) Q�;�n>2��VlQ.?��bȬ�0?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       ���g	��R����AU*�
w
discriminator_loss*a	   ���,?   ���,?      �?!   ���,?) ��yCj>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Ӿv��       ���g	\����AZ*�
w
discriminator_loss*a	   `�@?   `�@?      �?!   `�@?)@b2��>2d�\D�X=?���#@?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���>�       ���g	�pd����A_*�
w
discriminator_loss*a	   �;?   �;?      �?!   �;?) �����>2��%>��:?d�\D�X=?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��T�       ���g	�dm����Ad*�
w
discriminator_loss*a	   �7!?   �7!?      �?!   �7!?)@b�K�R>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       ���g	� v����Ai*�
w
discriminator_loss*a	    }�?    }�?      �?!    }�?) �P��=>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �D���       ���g	d�~����An*�
w
discriminator_loss*a	   `�?   `�?      �?!   `�?)@�Xyn^;>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        C���       ���g	d�����As*�
w
discriminator_loss*a	    )?    )?      �?!    )?) b��WN>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       ���g	<������Ax*�
w
discriminator_loss*a	    O!?    O!?      �?!    O!?) �I�PR>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Gk`�       ���g	� ����A}*�
w
discriminator_loss*a	   �]�?   �]�?      �?!   �]�?)���"�M>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �J�B�       �E��	H�����A�*�
w
discriminator_loss*a	    @�?    @�?      �?!    @�?) �P؁VH>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        s����       �E��	������A�*�
w
discriminator_loss*a	   @(?   @(?      �?!   @(?)���CW />2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        MWO��       �E��	y9����A�*�
w
discriminator_loss*a	   ��{4?   ��{4?      �?!   ��{4?) ɂ�8z>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        T ���       �E��	�3 ����A�*�
w
discriminator_loss*a	   `(?   `(?      �?!   `(?)@�@�6>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �@yE�       �E��	�W(����A�*�
w
discriminator_loss*a	    �e?    �e?      �?!    �e?)  ����2>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���%�       �E��	)�1����A�*�
w
discriminator_loss*a	   `�G$?   `�G$?      �?!   `�G$?)@�8�Y>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �o]��       �E��	��9����A�*�
w
discriminator_loss*a	   @.7?   @.7?      �?!   @.7?) �}�o0>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        C��       �E��	�đ¨��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)�Q�S}K>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        nW�5�       �E��	,��¨��A�*�
w
discriminator_loss*a	    �n?    �n?      �?!    �n?)  n���">2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��#�       �E��	lw�¨��A�*�
w
discriminator_loss*a	   @O�+?   @O�+?      �?!   @O�+?)�DL���g>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �-[�       �E��	�g�¨��A�*�
w
discriminator_loss*a	   @�>!?   @�>!?      �?!   @�>!?) �9d=�R>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �.���       �E��	On�¨��A�*�
w
discriminator_loss*a	   ��t3?   ��t3?      �?!   ��t3?) ���"�w>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �8�.�       �E��	1Y�¨��A�*�
w
discriminator_loss*a	   @)?   @)?      �?!   @)?) ќ}Ҝ4>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        r�       �E��	B<�¨��A�*�
w
discriminator_loss*a	    �� ?    �� ?      �?!    �� ?)  �-x3Q>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        *O��       �E��	^[�¨��A�*�
w
discriminator_loss*a	   `"�?   `"�?      �?!   `"�?) ��1�MH>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��       �E��	��Ĩ��A�*�
w
discriminator_loss*a	   ��e?   ��e?      �?!   ��e?) ���]3)>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]���       �E��	%�Ĩ��A�*�
w
discriminator_loss*a	   �(�D?   �(�D?      �?!   �(�D?) ��녛>2�T���C?a�$��{E?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        !=���       �E��	��%Ĩ��A�*�
w
discriminator_loss*a	   �ۍ(?   �ۍ(?      �?!   �ۍ(?) YUM>�b>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �[��       �E��	c�,Ĩ��A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ZS�>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        (K}�       �E��	��3Ĩ��A�*�
w
discriminator_loss*a	    �q?    �q?      �?!    �q?)  �\��.>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �>�&�       �E��	z�:Ĩ��A�*�
w
discriminator_loss*a	    �?    �?      �?!    �?) $6��i >26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	!�AĨ��A�*�
w
discriminator_loss*a	   �?   �?      �?!   �?) D!�9>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	��HĨ��A�*�
w
discriminator_loss*a	   �A�#?   �A�#?      �?!   �A�#?) $LKX>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        	L�E�       �E��	���Ũ��A�*�
w
discriminator_loss*a	   �*�?   �*�?      �?!   �*�?) �\(%"H>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        I��N�       �E��	V��Ũ��A�*�
w
discriminator_loss*a	   `bQ?   `bQ?      �?!   `bQ?)@�h��R7>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �K�?�       �E��	���Ũ��A�*�
w
discriminator_loss*a	   ���	?   ���	?      �?!   ���	?) 2G�My$>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��-��       �E��	܍�Ũ��A�*�
w
discriminator_loss*a	    �;?    �;?      �?!    �;?)  @�2>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �2��       �E��	Do�Ũ��A�*�
w
discriminator_loss*a	    �&?    �&?      �?!    �&?)@��%Ô^>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �w'Q�       �E��	W��Ũ��A�*�
w
discriminator_loss*a	   �ɜ?   �ɜ?      �?!   �ɜ?) �A��D>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        DD��       �E��	���Ũ��A�*�
w
discriminator_loss*a	   �H�)?   �H�)?      �?!   �H�)?)�dIkɽd>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �J��       �E��	���Ũ��A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) Dh�j�?>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�6+Ǩ��A�*�
w
discriminator_loss*a	    {�?    {�?      �?!    {�?) �Qx'�1>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Q�{�       �E��	�$3Ǩ��A�*�
w
discriminator_loss*a	   @��3?   @��3?      �?!   @��3?) ��x>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �c��       �E��	�+:Ǩ��A�*�
w
discriminator_loss*a	    d?    d?      �?!    d?) �X'�2>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ^����       �E��	�YAǨ��A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@ ˘3>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �hۣ�       �E��	ZeHǨ��A�*�
w
discriminator_loss*a	   �*
?   �*
?      �?!   �*
?) ���e%>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��T/�       �E��	�POǨ��A�*�
w
discriminator_loss*a	   ���2?   ���2?      �?!   ���2?) ��	��v>2��82?�u�w74?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        U�qA�       �E��	Q=VǨ��A�*�
w
discriminator_loss*a	   �ˊ ?   �ˊ ?      �?!   �ˊ ?) �҄KQ>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �N�       �E��	8]Ǩ��A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) H�,�@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��G��       �E��	���Ȩ��A�*�
w
discriminator_loss*a	    ?i&?    ?i&?      �?!    ?i&?) ؋!d_>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        _��E�       �E��	��Ȩ��A�*�
w
discriminator_loss*a	   ��[?   ��[?      �?!   ��[?) �{�2�>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	��Ȩ��A�*�
w
discriminator_loss*a	   �N�?   �N�?      �?!   �N�?) ����!>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���Q�       �E��	sx�Ȩ��A�*�
w
discriminator_loss*a	    +T?    +T?      �?!    +T?)@<D1�Y7>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]��P�       �E��	�`�Ȩ��A�*�
w
discriminator_loss*a	   ��[?   ��[?      �?!   ��[?) dT��0>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��C�       �E��	�^�Ȩ��A�*�
w
discriminator_loss*a	   @گ(?   @گ(?      �?!   @گ(?)�����c>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �{��       �E��	F�Ȩ��A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?)  ��P!>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        C�H��       �E��	W4�Ȩ��A�*�
w
discriminator_loss*a	    �:'?    �:'?      �?!    �:'?) �,/'�`>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        vB)��       �E��	@Bmʨ��A�*�
w
discriminator_loss*a	   @��=?   @��=?      �?!   @��=?)���p�:�>2d�\D�X=?���#@?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        H:�d�       �E��	�tʨ��A�*�
w
discriminator_loss*a	   @-�"?   @-�"?      �?!   @-�"?) ��w��U>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �9�d�       �E��	M�{ʨ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �TE�'>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��-��       �E��	��ʨ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) Df��:>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	b��ʨ��A�*�
w
discriminator_loss*a	   �
2?   �
2?      �?!   �
2?) 97��ht>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�ʨ��A�*�
w
discriminator_loss*a	   �$
?   �$
?      �?!   �$
?) ���8%>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        SԄ�       �E��	�ۗʨ��A�*�
w
discriminator_loss*a	   ``�?   ``�?      �?!   ``�?) A�&ToD>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	�̞ʨ��A�*�
w
discriminator_loss*a	    �<+?    �<+?      �?!    �<+?)  �N�.g>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �1R�       �E��	z)̨��A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?)  ��->2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��*g�       �E��	1�̨��A�*�
w
discriminator_loss*a	   @�$?   @�$?      �?!   @�$?)�,t�9�@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        w�i�       �E��	�� ̨��A�*�
w
discriminator_loss*a	   @ӽ?   @ӽ?      �?!   @ӽ?)��ָE�!>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        W}K��       �E��	�'̨��A�*�
w
discriminator_loss*a	    }(�>    }(�>      �?!    }(�>) Hh�l>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �We�       �E��	��.̨��A�*�
w
discriminator_loss*a	   �.�?   �.�?      �?!   �.�?)�L���u(>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        {(ab�       �E��	��5̨��A�*�
w
discriminator_loss*a	    \� ?    \� ?      �?!    \� ?)  �2}Q>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��       �E��	�<̨��A�*�
w
discriminator_loss*a	   @��)?   @��)?      �?!   @��)?)��4�e>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        W����       �E��	�C̨��A�*�
w
discriminator_loss*a	   @ׇ9?   @ׇ9?      �?!   @ׇ9?)�����^�>2uܬ�@8?��%>��:?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �,�       �E��	�S�ͨ��A�*�
w
discriminator_loss*a	   `�{?   `�{?      �?!   `�{?)@��]�<>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��A�       �E��	a�ͨ��A�*�
w
discriminator_loss*a	    T�?    T�?      �?!    T�?) �ܱvM>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	��ͨ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �dRW�->2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ~��       �E��	��ͨ��A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) ��>>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �e�o�       �E��	���ͨ��A�*�
w
discriminator_loss*a	   @�?   @�?      �?!   @�?)��pH�{#>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #����       �E��	E��ͨ��A�*�
w
discriminator_loss*a	   ���+?   ���+?      �?!   ���+?)�X�,h>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        {6"��       �E��	��ͨ��A�*�
w
discriminator_loss*a	   ��!?   ��!?      �?!   ��!?) Dj��S>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���)�       �E��	���ͨ��A�*�
w
discriminator_loss*a	   @��%?   @��%?      �?!   @��%?) �h|�\>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���R�       �E��	6uqϨ��A�*�
w
discriminator_loss*a	   �f�?   �f�?      �?!   �f�?) ٓo2E6>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �g���       �E��	�yϨ��A�*�
w
discriminator_loss*a	   �HE?   �HE?      �?!   �HE?) �f�>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        >G4$�       �E��	��Ϩ��A�*�
w
discriminator_loss*a	   ��q)?   ��q)?      �?!   ��q)?)��E��:d>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ky@u�       �E��	�a�Ϩ��A�*�
w
discriminator_loss*a	   ��s�>   ��s�>      �?!   ��s�>) ��&��=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��       �E��	�W�Ϩ��A�*�
w
discriminator_loss*a	    B+?    B+?      �?!    B+?) @�3<>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Q ��       �E��	�K�Ϩ��A�*�
w
discriminator_loss*a	   @��*?   @��*?      �?!   @��*?)��h;x�f>2I�I�)�(?�7Kaa+?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �� ��       �E��	�8�Ϩ��A�*�
w
discriminator_loss*a	   `�v?   `�v?      �?!   `�v?)@�w���>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	!�Ϩ��A�*�
w
discriminator_loss*a	   @c�?   @c�?      �?!   @c�?) �gV�D:>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        (��!�       �E��	ţ,Ѩ��A�*�
w
discriminator_loss*a	   @Є?   @Є?      �?!   @Є?)�@[�-->2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        u�TB�       �E��	�N4Ѩ��A�*�
w
discriminator_loss*a	   ��:?   ��:?      �?!   ��:?)��T�(>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	|E;Ѩ��A�*�
w
discriminator_loss*a	   �=�&?   �=�&?      �?!   �=�&?)�(���`>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        X���       �E��	PBѨ��A�*�
w
discriminator_loss*a	   �(�?   �(�?      �?!   �(�?) �p�}>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        9;P��       �E��	�PIѨ��A�*�
w
discriminator_loss*a	   `��?   `��?      �?!   `��?) �M��BA>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        J����       �E��	NPѨ��A�*�
w
discriminator_loss*a	   �0k?   �0k?      �?!   �0k?)@LIN�:>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	O-WѨ��A�*�
w
discriminator_loss*a	    �e?    �e?      �?!    �e?)@H�NI :>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��       �E��	J^Ѩ��A�*�
w
discriminator_loss*a	   �>V?   �>V?      �?!   �>V?) �o�F�">2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��<��       �E��	�A�Ҩ��A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?)�����A>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ѵR��       �E��	��Ҩ��A�*�
w
discriminator_loss*a	   �4�?   �4�?      �?!   �4�?) ���mqA>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	u��Ҩ��A�*�
w
discriminator_loss*a	    tA?    tA?      �?!    tA?)@��
Q�>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �N�I�       �E��	���Ҩ��A�*�
w
discriminator_loss*a	    �?    �?      �?!    �?) &ٷ�r@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��'��       �E��	��Ө��A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) ��j��D>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ym�X�       �E��	��Ө��A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) �}�O�?>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��c�       �E��	�Ө��A�*�
w
discriminator_loss*a	   �QH?   �QH?      �?!   �QH?)�ذmB>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        h��,�       �E��	l�Ө��A�*�
w
discriminator_loss*a	    /�?    /�?      �?!    /�?) 
�_1>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        $L
{�       �E��	���Ԩ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)@��5R9>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	p��Ԩ��A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) d�LCz�=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        tEn�       �E��	���Ԩ��A�*�
w
discriminator_loss*a	    �$?    �$?      �?!    �$?) l��Z>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        (�|��       �E��	���Ԩ��A�*�
w
discriminator_loss*a	   @�?   @�?      �?!   @�?)���ځ�+>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        vo��       �E��	���Ԩ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) "t�Vi+>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��?n�       �E��	���Ԩ��A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) ��	)#>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        |���       �E��	���Ԩ��A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �XҤ>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�z�Ԩ��A�*�
w
discriminator_loss*a	   �G�?   �G�?      �?!   �G�?) p}��B>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?         �t$�       �E��	b�֨��A�*�
w
discriminator_loss*a	   ��7?   ��7?      �?!   ��7?) ץ�tN>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        "c���       �E��	2��֨��A�*�
w
discriminator_loss*a	   @/�?   @/�?      �?!   @/�?) ��<<�>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�F�֨��A�*�
w
discriminator_loss*a	   @�?   @�?      �?!   @�?) �h�1�1>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �n��       �E��	���֨��A�*�
w
discriminator_loss*a	    a��>    a��>      �?!    a��>) ��PN>2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �{�       �E��	+"�֨��A�*�
w
discriminator_loss*a	   �"&�>   �"&�>      �?!   �"&�>) �R>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        	+��       �E��	��֨��A�*�
w
discriminator_loss*a	   �r ?   �r ?      �?!   �r ?) $��j�>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        rG�\�       �E��	��֨��A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) v�XD�!>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	���֨��A�*�
w
discriminator_loss*a	   �?   �?      �?!   �?) 8e�h*>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	���ب��A�*�
w
discriminator_loss*a	    :�?    :�?      �?!    :�?)  I?�^ >26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        [�M��       �E��	��ب��A�*�
w
discriminator_loss*a	   ��+?   ��+?      �?!   ��+?)�x��BB>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �L�A�       �E��	5��ب��A�*�
w
discriminator_loss*a	   ��I?   ��I?      �?!   ��I?)@�BG��2>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �N�S�       �E��	��ب��A�*�
w
discriminator_loss*a	   �ے?   �ے?      �?!   �ے?) �B���B>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        uY�)�       �E��	�#�ب��A�*�
w
discriminator_loss*a	   �jk?   �jk?      �?!   �jk?) r���.>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        e��       �E��	�g ٨��A�*�
w
discriminator_loss*a	   �x?   �x?      �?!   �x?) ��u[7A>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ]����       �E��	я٨��A�*�
w
discriminator_loss*a	   @9�?   @9�?      �?!   @9�?)�l�T�,>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��H�       �E��	�X٨��A�*�
w
discriminator_loss*a	    �>    �>      �?!    �>) ��,��>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        vE��       �E��	�l�ڨ��A�*�
w
discriminator_loss*a	    �2�>    �2�>      �?!    �2�>) �u� �=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	M;�ڨ��A�*�
w
discriminator_loss*a	   ��e ?   ��e ?      �?!   ��e ?)@��O��P>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        qO���       �E��	/�ڨ��A�*�
w
discriminator_loss*a	   ��R?   ��R?      �?!   ��R?)@j�]hj<>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�#�ڨ��A�*�
w
discriminator_loss*a	   �!u?   �!u?      �?!   �!u?) 1�5A>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        _���       �E��	<�ڨ��A�*�
w
discriminator_loss*a	   ��3?   ��3?      �?!   ��3?)@��S��>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        {��4�       �E��	u��ڨ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) X���">2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �2�+�       �E��	'�ڨ��A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)���z?C>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��X��       �E��	�3�ڨ��A�*�
w
discriminator_loss*a	    �55?    �55?      �?!    �55?)  !��|>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��Q�       �E��	���ܨ��A�*�
w
discriminator_loss*a	   �N�
?   �N�
?      �?!   �N�
?)��%Y�T&>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	?�ܨ��A�*�
w
discriminator_loss*a	   @�[�>   @�[�>      �?!   @�[�>)� ��>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �EF)�       �E��	tJ�ܨ��A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)����>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�7�ܨ��A�*�
w
discriminator_loss*a	   �8B?   �8B?      �?!   �8B?) I��ʝ2>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        3^�       �E��	�F�ܨ��A�*�
w
discriminator_loss*a	   @`��>   @`��>      �?!   @`��>)���'���=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �3�       �E��	���ܨ��A�*�
w
discriminator_loss*a	    F !?    F !?      �?!    F !?)@X��TR>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � �       �E��	�S�ܨ��A�*�
w
discriminator_loss*a	   ཯	?   ཯	?      �?!   ཯	?) �(�]�$>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��n��       �E��	�=�ܨ��A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)   �Q�>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        `!f�       �E��	���ި��A�*�
w
discriminator_loss*a	   �rd?   �rd?      �?!   �rd?)@.,��>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7�,�       �E��	��ި��A�*�
w
discriminator_loss*a	    Ϊ#?    Ϊ#?      �?!    Ϊ#?) @ܣ�,X>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        [�~�       �E��	)��ި��A�*�
w
discriminator_loss*a	   �L;?   �L;?      �?!   �L;?) )xwH�4>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        u�H�       �E��	���ި��A�*�
w
discriminator_loss*a	    �k?    �k?      �?!    �k?)  @��0>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        TQ�,�       �E��	���ި��A�*�
w
discriminator_loss*a	   �3�>   �3�>      �?!   �3�>) ��V#�>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        =��S�       �E��	���ި��A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@ ��d4>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	��ߨ��A�*�
w
discriminator_loss*a	   ��r?   ��r?      �?!   ��r?)@jUԩ>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Y��       �E��	�&ߨ��A�*�
w
discriminator_loss*a	   ��G?   ��G?      �?!   ��G?) b��.�.>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ܾ{�       �E��	������A�*�
w
discriminator_loss*a	   ��k?   ��k?      �?!   ��k?) �	y�">2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �
��       �E��	mn���A�*�
w
discriminator_loss*a	   ��<?   ��<?      �?!   ��<?)���1�.G>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��8�       �E��	�.���A�*�
w
discriminator_loss*a	   @G*?   @G*?      �?!   @G*?)����!Z.>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Ԩ�       M�	J����A*�
w
discriminator_loss*a	   `��$?   `��$?      �?!   `��$?)@����Z>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        5
��       ���g	}�����A*�
w
discriminator_loss*a	    f�?    f�?      �?!    f�?)  E�a/>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Zt�       ���g	�y����A
*�
w
discriminator_loss*a	   ��%?   ��%?      �?!   ��%?) �y1'>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ?�N�       ���g	�Z����A*�
w
discriminator_loss*a	    �!?    �!?      �?!    �!?) �?c1R>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        !�P��       ���g	A����A*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) ��x��&>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ~��g�       ���g	7:����A*�
w
discriminator_loss*a	    A%?    A%?      �?!    A%?)@�w�[>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � ���       ���g	�!���A*�
w
discriminator_loss*a	   @6?   @6?      �?!   @6?)��i��F>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ,]e!�       ���g	�%���A#*�
w
discriminator_loss*a	   ��S�>   ��S�>      �?!   ��S�>) ">	>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��{��       ���g	>	���A(*�
w
discriminator_loss*a	   �K�?   �K�?      �?!   �K�?)@r�KK�8>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        C7���       ���g	*�����A-*�
w
discriminator_loss*a	   ��8?   ��8?      �?!   ��8?) �Ӂ�>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��/�       ���g	������A2*�
w
discriminator_loss*a	   �,

?   �,

?      �?!   �,

?) �J(�0%>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �k���       ���g	2�����A7*�
w
discriminator_loss*a	   ��a?   ��a?      �?!   ��a?) ����<>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Ud�a�       ���g	�u����A<*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@�,�E�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        $�$��       ���g	2����AA*�
w
discriminator_loss*a	   �05?   �05?      �?!   �05?)@hS�q|>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        >�Ƀ�       ���g	�����AF*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)�@c����=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �;uH�       ���g		���AK*�
w
discriminator_loss*a	   �D�?   �D�?      �?!   �D�?) �2<^I@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        )�O��       ���g	�����AP*�
w
discriminator_loss*a	   �Y�?   �Y�?      �?!   �Y�?) RZ��G>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        6�`T�       ���g	�����AU*�
w
discriminator_loss*a	   �d�>   �d�>      �?!   �d�>)�0|�[0	>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        uꢠ�       ���g	����AZ*�
w
discriminator_loss*a	   ��"?   ��"?      �?!   ��"?) �tO��U>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?         �K�       ���g	[����A_*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �o�4/>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ˉG�       ���g	p+$���Ad*�
w
discriminator_loss*a	   �_�?   �_�?      �?!   �_�?) mR=�;>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        }۫��       ���g	��,���Ai*�
w
discriminator_loss*a	   `(�?   `(�?      �?!   `(�?)@�c]3>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ix�       ���g	�4���An*�
w
discriminator_loss*a	    �c?    �c?      �?!    �c?) ��y��L>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        2�%"�       ���g	~+;���As*�
w
discriminator_loss*a	    A!8?    A!8?      �?!    A!8?) �2�>2��%�V6?uܬ�@8?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �\N��       ���g	�B���Ax*�
w
discriminator_loss*a	   ��(?   ��(?      �?!   ��(?) ���A�>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �SS�       ���g	ĝR���A}*�
w
discriminator_loss*a	   �"�?   �"�?      �?!   �"�?)��I~�C>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��~�       �E��	=[���A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) �_�1?>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �֣�       �E��	��d���A�*�
w
discriminator_loss*a	   �l��>   �l��>      �?!   �l��>) �ﴆ��=2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �KT��       �E��	��l���A�*�
w
discriminator_loss*a	   �� ?   �� ?      �?!   �� ?)@�r�F_>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	��v���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �FCW>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��%J�       �E��	�@~���A�*�
w
discriminator_loss*a	   �� ?   �� ?      �?!   �� ?)@�����>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Se�#�       �E��	�ԅ���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) ��yI>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �
�k�       �E��	5����A�*�
w
discriminator_loss*a	   �t�>   �t�>      �?!   �t�>) ²M�>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        3�5��       �E��	�۬���A�*�
w
discriminator_loss*a	   �ot?   �ot?      �?!   �ot?) a� 1!>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �|���       �E��	JT����A�*�
w
discriminator_loss*a	    w�?    w�?      �?!    w�?) ����?>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        1�H�       �E��	o����A�*�
w
discriminator_loss*a	   `��?   `��?      �?!   `��?)@�T��>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	<�����A�*�
w
discriminator_loss*a	    �2?    �2?      �?!    �2?)  D��>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        h-B�       �E��	#F����A�*�
w
discriminator_loss*a	   �K?   �K?      �?!   �K?)�P�3�@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	1?����A�*�
w
discriminator_loss*a	   �U�+?   �U�+?      �?!   �U�+?) ��g>2�7Kaa+?��VlQ.?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        &֌H�       �E��	H����A�*�
w
discriminator_loss*a	   @O�?   @O�?      �?!   @O�?) ��`|#>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        t`J��       �E��	Uf����A�*�
w
discriminator_loss*a	   �@}?   �@}?      �?!   �@}?) 	�4�]>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        LN�\�       �E��	#l�����A�*�
w
discriminator_loss*a	    �1?    �1?      �?!    �1?) d���@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��|v�       �E��	�M���A�*�
w
discriminator_loss*a	   �|-?   �|-?      �?!   �|-?)@��t�<>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        k&���       �E��	��	���A�*�
w
discriminator_loss*a	    z}�>    z}�>      �?!    z}�>) ��0>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ڧ�_�       �E��	����A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) K/`6>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �}b��       �E��	�E���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?)�x���J>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��j��       �E��	<H���A�*�
w
discriminator_loss*a	    �?    �?      �?!    �?) R�:3>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��q�       �E��	kU&���A�*�
w
discriminator_loss*a	    6��>    6��>      �?!    6��>) ��жX>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        '(V�       �E��	��-���A�*�
w
discriminator_loss*a	    vh?    vh?      �?!    vh?) @f��-5>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �8a�       �E��	B*4���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)��&�]>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        +��-�       �E��	n�<���A�*�
w
discriminator_loss*a	    �g�>    �g�>      �?!    �g�>) �z:J>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        H�G��       �E��	t�D���A�*�
w
discriminator_loss*a	   `�?   `�?      �?!   `�?) �����(>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �`���       �E��	x�K���A�*�
w
discriminator_loss*a	   ��9 ?   ��9 ?      �?!   ��9 ?)@���tP>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �\s|�       �E��	֪S���A�*�
w
discriminator_loss*a	   �� ?   �� ?      �?!   �� ?)@��4v>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �;���       �E��	v�Z���A�*�
w
discriminator_loss*a	    c?    c?      �?!    c?) Hb�`�&>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        B
���       �E��	2b���A�*�
w
discriminator_loss*a	   @/�?   @/�?      �?!   @/�?) ���n�>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �,4k�       �E��	bHi���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) iC��>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ɂF �       �E��	�N����A�*�
w
discriminator_loss*a	    }(�>    }(�>      �?!    }(�>) B���<�=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        qv�7�       �E��	f
����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�l��T�>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        E���       �E��	������A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@ʯ����=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���q�       �E��	c�����A�*�
w
discriminator_loss*a	   `��?   `��?      �?!   `��?) k:;��'>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��`��       �E��	�#����A�*�
w
discriminator_loss*a	    �r?    �r?      �?!    �r?) �,a":>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        LTr��       �E��	�W����A�*�
w
discriminator_loss*a	   �B�?   �B�?      �?!   �B�?) d�&m[>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �f�n�       �E��	\ ����A�*�
w
discriminator_loss*a	   `+D�>   `+D�>      �?!   `+D�>)@�%����=2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        *\��       �E��	[e����A�*�
w
discriminator_loss*a	   @B�>   @B�>      �?!   @B�>) � ^�-�=2jqs&\��>��~]�[�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	#������A�*�
w
discriminator_loss*a	    �1?    �1?      �?!    �1?) ���>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	�3�����A�*�
w
discriminator_loss*a	   @�G�>   @�G�>      �?!   @�G�>)����A>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	P�����A�*�
w
discriminator_loss*a	    \�?    \�?      �?!    \�?)  ˾�:>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ^\�f�       �E��	@-�����A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)����E>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        4�h��       �E��	X�����A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@l�0C8>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        L�!��       �E��	�������A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?)@�2��6>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        	�!��       �E��	������A�*�
w
discriminator_loss*a	   @~?   @~?      �?!   @~?)���� >26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	̕�����A�*�
w
discriminator_loss*a	    �?    �?      �?!    �?) ���g>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � ���       �E��	k�����A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) e��!�>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	�����A�*�
w
discriminator_loss*a	   @;�?   @;�?      �?!   @;�?)��a�	E >26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ,LX�       �E��	4m����A�*�
w
discriminator_loss*a	    �u?    �u?      �?!    �u?)  �j�):>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �G!��       �E��	�&����A�*�
w
discriminator_loss*a	    �7�>    �7�>      �?!    �7�>)  �y1�>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        l��M�       �E��	)�-����A�*�
w
discriminator_loss*a	    �?    �?      �?!    �?) @����;>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �� �       �E��	Qj5����A�*�
w
discriminator_loss*a	   `�� ?   `�� ?      �?!   `�� ?)@~�A��>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        X��       �E��	'�<����A�*�
w
discriminator_loss*a	    $��>    $��>      �?!    $��>)  Q�g�=2�ѩ�-�>���%�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        c�W��       �E��	�D����A�*�
w
discriminator_loss*a	    L��>    L��>      �?!    L��>) �����>2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	*�h����A�*�
w
discriminator_loss*a	   �K��>   �K��>      �?!   �K��>) "����>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        c�՘�       �E��	�Sp����A�*�
w
discriminator_loss*a	   �Z�?   �Z�?      �?!   �Z�?) ��v�>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ^ۆ��       �E��	c�w����A�*�
w
discriminator_loss*a	   �4�?   �4�?      �?!   �4�?)��W)j >26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        mׄn�       �E��	�~����A�*�
w
discriminator_loss*a	    (�?    (�?      �?!    (�?)  �Ƞ�/>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        J�O�       �E��	�B�����A�*�
w
discriminator_loss*a	   @њ?   @њ?      �?!   @њ?) ���0�5>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        K�[u�       �E��	}������A�*�
w
discriminator_loss*a	   ��u?   ��u?      �?!   ��u?) ����>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ܈e��       �E��	oV�����A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@"u���=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��a��       �E��	�������A�*�
w
discriminator_loss*a	    Bi�>    Bi�>      �?!    Bi�>) �z6��>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	������A�*�
w
discriminator_loss*a	    G-?    G-?      �?!    G-?) � �p>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        )���       �E��	������A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �9���+>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��       �E��	������A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)  �ނ>2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        0����       �E��	�L�����A�*�
w
discriminator_loss*a	   `�5?   `�5?      �?!   `�5?)@�w�y�{>2�u�w74?��%�V6?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        	�o��       �E��	�G�����A�*�
w
discriminator_loss*a	   �uX�>   �uX�>      �?!   �uX�>) ]�`��=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Cz\m�       �E��	_������A�*�
w
discriminator_loss*a	   @ ?   @ ?      �?!   @ ?) 9��H,>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ī��       �E��	������A�*�
w
discriminator_loss*a	   �k��>   �k��>      �?!   �k��>) "�����=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        [9v��       �E��	:+�����A�*�
w
discriminator_loss*a	   �v�?   �v�?      �?!   �v�?) �Wy�I>2�.�?ji6�9�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �f�       �E��	��E����A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) J1�fe>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��$��       �E��	GO����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �x���=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �u��       �E��	�V����A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@F]m[�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Ku��       �E��	�]����A�*�
w
discriminator_loss*a	   �N?   �N?      �?!   �N?) iQ��J>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        G0�|�       �E��	[�d����A�*�
w
discriminator_loss*a	    @�"?    @�"?      �?!    @�"?)   �8�U>2�S�F !?�[^:��"?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Xn��       �E��	�l����A�*�
w
discriminator_loss*a	   �7��>   �7��>      �?!   �7��>) ������=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���V�       �E��	{s����A�*�
w
discriminator_loss*a	   �S ?   �S ?      �?!   �S ?) �ٯ�� >26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        F����       �E��	��z����A�*�
w
discriminator_loss*a	    pd'?    pd'?      �?!    pd'?)  �=�a>2+A�F�&?I�I�)�(?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        3Oz0�       �E��	�t� ���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) �,�)�>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��NC�       �E��	�*� ���A�*�
w
discriminator_loss*a	   �� ?   �� ?      �?!   �� ?) ia�j�>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        6���       �E��	sY� ���A�*�
w
discriminator_loss*a	    n?    n?      �?!    n?)  ��'!>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        .��       �E��	hn� ���A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) ��LCC>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Aج��       �E��	
�� ���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) �r�!*�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        CyI��       �E��	#�� ���A�*�
w
discriminator_loss*a	   �0�?   �0�?      �?!   �0�?)@�'K�5>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �}
��       �E��	@g ���A�*�
w
discriminator_loss*a	    �0$?    �0$?      �?!    �0$?)@(<{Y>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        )	�       �E��	\����A�*�
w
discriminator_loss*a	   @I?   @I?      �?!   @I?)��c6�">2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �h��       �E��	*]���A�*�
w
discriminator_loss*a	   ��Z?   ��Z?      �?!   ��Z?)��a^C!>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �T�       �E��	��d���A�*�
w
discriminator_loss*a	   �z��>   �z��>      �?!   �z��>)@�G6�z�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �q0�       �E��	5m���A�*�
w
discriminator_loss*a	   `aC�>   `aC�>      �?!   `aC�>) O���=2�ѩ�-�>���%�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        UŌ�       �E��	b�t���A�*�
w
discriminator_loss*a	   ��%?   ��%?      �?!   ��%?) Xi�>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        zT�       �E��	�Q}���A�*�
w
discriminator_loss*a	   @�Z?   @�Z?      �?!   @�Z?) �S�i>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �!X�       �E��	Oe����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) B�|z >2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ޹�c�       �E��	������A�*�
w
discriminator_loss*a	   �s��>   �s��>      �?!   �s��>) a5���=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ړN��       �E��	oQ����A�*�
w
discriminator_loss*a	    �3�>    �3�>      �?!    �3�>) @��g�=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        d�K7�       �E��	�B����A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?)  ҫ�c >26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��D��       �E��	�b����A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) H�{��=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Q>Q�       �E��	�����A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) )���=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ي��       �E��	�>���A�*�
w
discriminator_loss*a	   @N^?   @N^?      �?!   @N^?)�Xw�&)>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �|��       �E��	�����A�*�
w
discriminator_loss*a	   @	��>   @	��>      �?!   @	��>) Y��w��=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        PY�A�       �E��	&l���A�*�
w
discriminator_loss*a	    8k�>    8k�>      �?!    8k�>)  �>��>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        X,R��       �E��	x%���A�*�
w
discriminator_loss*a	    `�>    `�>      �?!    `�>) 	�8>2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        z�]�       �E��	c�,���A�*�
w
discriminator_loss*a	   @1p�>   @1p�>      �?!   @1p�>)��%�*�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        0r.��       �E��	��y���A�*�
w
discriminator_loss*a	   �K8?   �K8?      �?!   �K8?) ���]q0>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ԅ���       �E��	�o����A�*�
w
discriminator_loss*a	   �K?   �K?      �?!   �K?)@t@�p>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �#���       �E��	E�����A�*�
w
discriminator_loss*a	    ?    ?      �?!    ?) ��~��>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Ga��       �E��	$ܐ���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �(`>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ^���       �E��	������A�*�
w
discriminator_loss*a	   ���	?   ���	?      �?!   ���	?) �r�$>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        M���       �E��	����A�*�
w
discriminator_loss*a	   @3��>   @3��>      �?!   @3��>) )t��=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        l�Q��       �E��	@O����A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) ��*8�>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        SEJ�       �E��	4�����A�*�
w
discriminator_loss*a	   �
�
?   �
�
?      �?!   �
�
?) �M�B&>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Z����       �E��	%��
���A�*�
w
discriminator_loss*a	    �s�>    �s�>      �?!    �s�>)  ����=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �cR0�       �E��	�i�
���A�*�
w
discriminator_loss*a	   ��_�>   ��_�>      �?!   ��_�>)@Ǝ��=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �v��       �E��	� ���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) @�\�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        4���       �E��	����A�*�
w
discriminator_loss*a	    p3?    p3?      �?!    p3?)  ���k.>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �=v��       �E��	�=���A�*�
w
discriminator_loss*a	   @a�?   @a�?      �?!   @a�?)��ˏ�J >26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �q���       �E��	3	���A�*�
w
discriminator_loss*a	   ��|?   ��|?      �?!   ��|?)@��,';>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	�7���A�*�
w
discriminator_loss*a	    R�?    R�?      �?!    R�?)  ҏ�A>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �fL�       �E��	r\%���A�*�
w
discriminator_loss*a	   `�5�>   `�5�>      �?!   `�5�>) �#��>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	 V����A�*�
w
discriminator_loss*a	    !0�>    !0�>      �?!    !0�>) "S�e�=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	�	����A�*�
w
discriminator_loss*a	    �A?    �A?      �?!    �A?)@ǅ�0>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �� �       �E��	�I����A�*�
w
discriminator_loss*a	   ��l�>   ��l�>      �?!   ��l�>) �$�>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��]��       �E��	n�����A�*�
w
discriminator_loss*a	   �]��>   �]��>      �?!   �]��>) d�"w��=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %�O��       �E��	7�����A�*�
w
discriminator_loss*a	   @F?   @F?      �?!   @F?) qDG�;>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ǖr��       �E��	���A�*�
w
discriminator_loss*a	   �=�>   �=�>      �?!   �=�>)�����~>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �`#8�       �E��	������A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)��V���>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        RcQ�       �E��	������A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?)@ 翔Q8>2�5�i}1?�T7��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        2�S�       �E��	��a���A�*�
w
discriminator_loss*a	   `]� ?   `]� ?      �?!   `]� ?)@��d*K>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        t����       �E��	�Uj���A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) ��&�3>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        G�CS�       �E��	��r���A�*�
w
discriminator_loss*a	   �+��>   �+��>      �?!   �+��>)�����=2�iD*L��>E��a�W�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �W�       �E��	��z���A�*�
w
discriminator_loss*a	   �:��>   �:��>      �?!   �:��>)@�R��K�=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �I��       �E��	nO����A�*�
w
discriminator_loss*a	   �\� ?   �\� ?      �?!   �\� ?)@6D!��>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        U;j�       �E��	dJ����A�*�
w
discriminator_loss*a	   ��4�>   ��4�>      �?!   ��4�>) ��S#�
>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7t���       �E��	+�����A�*�
w
discriminator_loss*a	   �R{�>   �R{�>      �?!   �R{�>) dfٖ�=2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���o�       �E��	�̞���A�*�
w
discriminator_loss*a	   `*n?   `*n?      �?!   `*n?)@:��2>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �eQ�       �E��	�yz���A�*�
w
discriminator_loss*a	   ��H�>   ��H�>      �?!   ��H�>) ��{���=2�ѩ�-�>���%�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        UF��       �E��	P����A�*�
w
discriminator_loss*a	    �#?    �#?      �?!    �#?)@����V>2�[^:��"?U�4@@�$?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��C��       �E��	�6����A�*�
w
discriminator_loss*a	    �j?    �j?      �?!    �j?) �I�s}G>2��ڋ?�.�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �w���       �E��	�a����A�*�
w
discriminator_loss*a	   �0y�>   �0y�>      �?!   �0y�>) ��s%>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �7��       �E��	k(����A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) Be�aR>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        oc��       �E��	�4����A�*�
w
discriminator_loss*a	    
?    
?      �?!    
?)@H�
�V4>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��+�       �E��	=����A�*�
w
discriminator_loss*a	    �U?    �U?      �?!    �U?) �d�2>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        _��       �E��	潸���A�*�
w
discriminator_loss*a	   �J5�>   �J5�>      �?!   �J5�>) 2�ҙo�=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��       �E��	J�K���A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?) ����!>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��`�       �E��	c2T���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�jк��=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��8��       �E��	�9\���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) H�vd[>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        <$��       M�	(c���A*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) �UujO#>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �9w�       ���g	����A*�
w
discriminator_loss*a	   �ۉ�>   �ۉ�>      �?!   �ۉ�>) Ȕ4W�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        .��[�       ���g	����A
*�
w
discriminator_loss*a	    K:�>    K:�>      �?!    K:�>) ȏж�>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        V����       ���g	�M���A*�
w
discriminator_loss*a	   �?��>   �?��>      �?!   �?��>) �����=2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Ϥ�       ���g	��(���A*�
w
discriminator_loss*a	    k?    k?      �?!    k?)@<��>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Q�c�       ���g	�v0���A*�
w
discriminator_loss*a	   �]��>   �]��>      �?!   �]��>) d�6��=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �:�:�       ���g	�J8���A*�
w
discriminator_loss*a	   @N��>   @N��>      �?!   @N��>)�X78qt�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        f����       ���g	�A���A#*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)  ��[A�=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��h�       ���g	��H���A(*�
w
discriminator_loss*a	   `�r�>   `�r�>      �?!   `�r�>) �#8��>2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �q���       ���g	H'���A-*�
w
discriminator_loss*a	   �D��>   �D��>      �?!   �D��>) +�+���=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��o�       ���g	O�.���A2*�
w
discriminator_loss*a	   ��c ?   ��c ?      �?!   ��c ?) �|p�>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        fVC=�       ���g	�p7���A7*�
w
discriminator_loss*a	   �m�>   �m�>      �?!   �m�>) d�4Gx�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        F�0�       ���g	H?���A<*�
w
discriminator_loss*a	   �.�>   �.�>      �?!   �.�>) ���\�=2K+�E���>jqs&\��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        /vP�       ���g	��F���AA*�
w
discriminator_loss*a	   ��b?   ��b?      �?!   ��b?) 1�y��>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ˄Q�       ���g	��M���AF*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) dP�=�)>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       ���g	��U���AK*�
w
discriminator_loss*a	    v�>    v�>      �?!    v�>) �ە^��=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���t�       ���g	�_���AP*�
w
discriminator_loss*a	    (�>    (�>      �?!    (�>)  ����>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �B��       ���g	������AU*�
w
discriminator_loss*a	   �.?   �.?      �?!   �.?) �z�
<>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��29�       ���g	"�����AZ*�
w
discriminator_loss*a	   �\��>   �\��>      �?!   �\��>) Ć`ì�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        }�P��       ���g	+����A_*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)@
����=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        w���       ���g	���Ad*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)�Xj0�W>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        fz��       ���g	�����Ai*�
w
discriminator_loss*a	   ��M�>   ��M�>      �?!   ��M�>) 
y��=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �3��       ���g	8����An*�
w
discriminator_loss*a	   �c%�>   �c%�>      �?!   �c%�>) Ě�#��=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        q�se�       ���g	�����As*�
w
discriminator_loss*a	   ��:�>   ��:�>      �?!   ��:�>) ��!�>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��|S�       ���g	��"���Ax*�
w
discriminator_loss*a	   �E��>   �E��>      �?!   �E��>) �,1��=2E��a�W�>�ѩ�-�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �x��       ���g	���!���A}*�
w
discriminator_loss*a	   ��j?   ��j?      �?!   ��j?) D�t�>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �(3�       �E��	���!���A�*�
w
discriminator_loss*a	    �q?    �q?      �?!    �q?)@$�>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �6t�       �E��	EP�!���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ʕ\��=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �E���       �E��	�o�!���A�*�
w
discriminator_loss*a	   ��E�>   ��E�>      �?!   ��E�>)@���6�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        R$^��       �E��	an�!���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) <�a��=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ֣`�       �E��	ك�!���A�*�
w
discriminator_loss*a	   @6�>   @6�>      �?!   @6�>)�����G�=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        =�	�       �E��	>��!���A�*�
w
discriminator_loss*a	   `wh�>   `wh�>      �?!   `wh�>) S�S���=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        :���       �E��	Y��!���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) ����@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�l�$���A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)�����>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �E/=�       �E��	/��$���A�*�
w
discriminator_loss*a	   �� ?   �� ?      �?!   �� ?) �m�],>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ,h<�       �E��	�:�$���A�*�
w
discriminator_loss*a	   � ��>   � ��>      �?!   � ��>) ��~gc�=2;�"�q�>['�?��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	��$���A�*�
w
discriminator_loss*a	   �6h�>   �6h�>      �?!   �6h�>) Ҝ²7�=2E��a�W�>�ѩ�-�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        +����       �E��	�`�$���A�*�
w
discriminator_loss*a	   � L�>   � L�>      �?!   � L�>)@��0��=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �� F�       �E��	nA�$���A�*�
w
discriminator_loss*a	   �	�?   �	�?      �?!   �	�?) �U,��>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        @��k�       �E��	�$���A�*�
w
discriminator_loss*a	   �$� ?   �$� ?      �?!   �$� ?)@��U&>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        i���       �E��	�j�$���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) ��`�(>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �S�       �E��	u�l'���A�*�
w
discriminator_loss*a	   �Q��>   �Q��>      �?!   �Q��>)��X���=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	��u'���A�*�
w
discriminator_loss*a	    ̟�>    ̟�>      �?!    ̟�>)@0(ǳ9�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        sMa��       �E��	�}'���A�*�
w
discriminator_loss*a	   �:�>   �:�>      �?!   �:�>) �Ŗ���=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        y,���       �E��	`�'���A�*�
w
discriminator_loss*a	   �"��>   �"��>      �?!   �"��>) y�0|�=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��O��       �E��	�U�'���A�*�
w
discriminator_loss*a	   `_��>   `_��>      �?!   `_��>)@�\b��=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Iܴ�       �E��	�{�'���A�*�
w
discriminator_loss*a	   ��>   ��>      �?!   ��>)  �\B�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��J�       �E��	ߜ'���A�*�
w
discriminator_loss*a	   �x��>   �x��>      �?!   �x��>) ����	>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �H�a�       �E��	e^�'���A�*�
w
discriminator_loss*a	    ��?    ��?      �?!    ��?)@X��~64>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��%�       �E��	g�M*���A�*�
w
discriminator_loss*a	   �Æ�>   �Æ�>      �?!   �Æ�>) ��c��=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �[s��       �E��	P�U*���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ��H>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        .�;>�       �E��	Z�\*���A�*�
w
discriminator_loss*a	   �\�0?   �\�0?      �?!   �\�0?) �����q>2��bȬ�0?��82?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        XS=�       �E��	�d*���A�*�
w
discriminator_loss*a	    �x�>    �x�>      �?!    �x�>)@���S�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        
����       �E��	
�k*���A�*�
w
discriminator_loss*a	   `x+
?   `x+
?      �?!   `x+
?) �F��f%>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ڰs�       �E��	�Ds*���A�*�
w
discriminator_loss*a	   `�F ?   `�F ?      �?!   `�F ?)@�0�Y�>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        R���       �E��	�Qz*���A�*�
w
discriminator_loss*a	   �X��>   �X��>      �?!   �X��>) �yCRZ�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � Ѣ�       �E��	a�*���A�*�
w
discriminator_loss*a	   @1&�>   @1&�>      �?!   @1&�>)��Cuw9�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��;G�       �E��	2-���A�*�
w
discriminator_loss*a	   �V�>   �V�>      �?!   �V�>)@D��.�=2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        :i���       �E��	w�:-���A�*�
w
discriminator_loss*a	   ��J�>   ��J�>      �?!   ��J�>)�$Q�p�=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	��A-���A�*�
w
discriminator_loss*a	   �Ɯ�>   �Ɯ�>      �?!   �Ɯ�>)@��j�?�=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	P�I-���A�*�
w
discriminator_loss*a	   @�I�>   @�I�>      �?!   @�I�>) q��@�=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        
��       �E��	K�P-���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) �k�N,>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �˾��       �E��	�X-���A�*�
w
discriminator_loss*a	   `�n�>   `�n�>      �?!   `�n�>)@���ߵ�=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        8���       �E��	f_-���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) �]�>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        KP�P�       �E��	,�f-���A�*�
w
discriminator_loss*a	   @y��>   @y��>      �?!   @y��>)�l/ ��=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �+��       �E��	b�>0���A�*�
w
discriminator_loss*a	   �i�
?   �i�
?      �?!   �i�
?) L~A��&>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        1K�       �E��	�G0���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ���=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �S�*�       �E��	��N0���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) �X��f>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        E;M�       �E��	G�U0���A�*�
w
discriminator_loss*a	   �҂ ?   �҂ ?      �?!   �҂ ?) dq��	>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	+�\0���A�*�
w
discriminator_loss*a	   �,7�>   �,7�>      �?!   �,7�>)@vHg�!�=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �'-#�       �E��	�d0���A�*�
w
discriminator_loss*a	   �z
?   �z
?      �?!   �z
?) O��C�%>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        M�)��       �E��	x�k0���A�*�
w
discriminator_loss*a	   ��S�>   ��S�>      �?!   ��S�>) l��>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �M+��       �E��	�fs0���A�*�
w
discriminator_loss*a	    u�>    u�>      �?!    u�>) ��aGC�=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �-��       �E��	8�>3���A�*�
w
discriminator_loss*a	    �S�>    �S�>      �?!    �S�>) 
�A��=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        t2���       �E��	W�G3���A�*�
w
discriminator_loss*a	   �%n�>   �%n�>      �?!   �%n�>) ��CA��=2�ѩ�-�>���%�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        � n��       �E��	��O3���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)��+�|>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        6h1�       �E��	��V3���A�*�
w
discriminator_loss*a	   `O��>   `O��>      �?!   `O��>) �HT�1�=2�iD*L��>E��a�W�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ?}��       �E��	l�^3���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �e�>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	j�e3���A�*�
w
discriminator_loss*a	   �)�?   �)�?      �?!   �)�?) %�\�/>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �m�       �E��	��l3���A�*�
w
discriminator_loss*a	   ��K�>   ��K�>      �?!   ��K�>)��}�	>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        }l3��       �E��	��s3���A�*�
w
discriminator_loss*a	   �D��>   �D��>      �?!   �D��>) D�ۆ�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �F��       �E��	-!j6���A�*�
w
discriminator_loss*a	    +�>    +�>      �?!    +�>)@<����=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��$�       �E��	�Vs6���A�*�
w
discriminator_loss*a	   `��?   `��?      �?!   `��?) ��%3C@>2�T7��?�vV�R9?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �`{�       �E��	Y{6���A�*�
w
discriminator_loss*a	   ���?   ���?      �?!   ���?) D˹�>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �p�       �E��	0��6���A�*�
w
discriminator_loss*a	   �U�>   �U�>      �?!   �U�>) ���o�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        R%�%�       �E��	�.�6���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)  ��X�=2�iD*L��>E��a�W�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �Q��       �E��	���6���A�*�
w
discriminator_loss*a	   �4��>   �4��>      �?!   �4��>)@����=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        j�M�       �E��	*�6���A�*�
w
discriminator_loss*a	   �1 �>   �1 �>      �?!   �1 �>) $��Ș�=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ӂ���       �E��	U�6���A�*�
w
discriminator_loss*a	   �l��>   �l��>      �?!   �l��>) )ۛiJ�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        YNcP�       �E��	�E�9���A�*�
w
discriminator_loss*a	   @ש�>   @ש�>      �?!   @ש�>) �w����=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	iД9���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) B����>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	B�9���A�*�
w
discriminator_loss*a	    
�>    
�>      �?!    
�>)@�ƛ2�=2K+�E���>jqs&\��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���y�       �E��	4v�9���A�*�
w
discriminator_loss*a	   `pe�>   `pe�>      �?!   `pe�>) ��$�'�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ȩP�       �E��	ڕ�9���A�*�
w
discriminator_loss*a	   @�a�>   @�a�>      �?!   @�a�>)��C�^,�=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �яM�       �E��	{��9���A�*�
w
discriminator_loss*a	   `��?   `��?      �?!   `��?)@fM[t6>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        l?���       �E��	1ݸ9���A�*�
w
discriminator_loss*a	   `Ӑ?   `Ӑ?      �?!   `Ӑ?)@v��H3>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        k�TV�       �E��	+��9���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ��f�=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	���<���A�*�
w
discriminator_loss*a	     K�>     K�>      �?!     K�>)@�l	�C�=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �zK�       �E��	�c�<���A�*�
w
discriminator_loss*a	    Q?    Q?      �?!    Q?) �u�>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���D�       �E��	�O�<���A�*�
w
discriminator_loss*a	   @�D�>   @�D�>      �?!   @�D�>)�pԯ��>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        b�qE�       �E��	͟�<���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) �?6�_�=2�����>
�/eq
�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��:��       �E��	�1�<���A�*�
w
discriminator_loss*a	    }�?    }�?      �?!    }�?) B�k(>2f�ʜ�7
?>h�'�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        at��       �E��	2��<���A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) iT���=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	n��<���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@Fo
zտ=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        (}4$�       �E��	��<���A�*�
w
discriminator_loss*a	   �~K�>   �~K�>      �?!   �~K�>) $� �=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        "�cJ�       �E��	q`�?���A�*�
w
discriminator_loss*a	   ��u?   ��u?      �?!   ��u?)@�����>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �n�       �E��	�U�?���A�*�
w
discriminator_loss*a	   �5l�>   �5l�>      �?!   �5l�>)@��Lol�=2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ^~�|�       �E��	��?���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>)@Ĕ�2��=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �1��       �E��	�I�?���A�*�
w
discriminator_loss*a	   `y��>   `y��>      �?!   `y��>) _JP[�>2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �7��       �E��	��?���A�*�
w
discriminator_loss*a	    �`?    �`?      �?!    �`?) ���~!>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �NCC�       �E��	��?���A�*�
w
discriminator_loss*a	    ?    ?      �?!    ?)@�x3	P4>2��d�r?�5�i}1?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��3��       �E��	�F�?���A�*�
w
discriminator_loss*a	    y��>    y��>      �?!    y��>) ����=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��m�       �E��	v|�?���A�*�
w
discriminator_loss*a	   @�'�>   @�'�>      �?!   @�'�>) 1�g��=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �L"��       �E��	�C���A�*�
w
discriminator_loss*a	     �?     �?      �?!     �?)   >�C>2�vV�R9?��ڋ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �GS|�       �E��	+n&C���A�*�
w
discriminator_loss*a	    ^��>    ^��>      �?!    ^��>)  �>��=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?         �� �       �E��	r�-C���A�*�
w
discriminator_loss*a	    �~�>    �~�>      �?!    �~�>)@\����=2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �m�u�       �E��	G�5C���A�*�
w
discriminator_loss*a	   @�t�>   @�t�>      �?!   @�t�>)�����=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��6�       �E��	�c=C���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) Xb��=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ߨ_�       �E��	d�EC���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) ���/�=2��~���>�XQ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �~�       �E��	�NC���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>)� ���=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �֔[�       �E��	<:UC���A�*�
w
discriminator_loss*a	   �]u�>   �]u�>      �?!   �]u�>) 2�<���=2�iD*L��>E��a�W�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Gm�_�       �E��	��F���A�*�
w
discriminator_loss*a	   ��X�>   ��X�>      �?!   ��X�>) &���=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %��Y�       �E��	�C�F���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)�H.��=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        D���       �E��	�H�F���A�*�
w
discriminator_loss*a	   �+|?   �+|?      �?!   �+|?)��K�*+>2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��3��       �E��	���F���A�*�
w
discriminator_loss*a	   �A@�>   �A@�>      �?!   �A@�>) �^���=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �T���       �E��	ߠ�F���A�*�
w
discriminator_loss*a	    �;�>    �;�>      �?!    �;�>)  >���
>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �}p��       �E��	Ȩ�F���A�*�
w
discriminator_loss*a	   ��?   ��?      �?!   ��?) XeO!>26�]��?����?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �n$�       �E��	3׬F���A�*�
w
discriminator_loss*a	    %	?    %	?      �?!    %	?)@$t#2>2x?�x�?��d�r?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �0���       �E��	�F���A�*�
w
discriminator_loss*a	   �$��>   �$��>      �?!   �$��>)@�'���=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��n��       �E��	�\�I���A�*�
w
discriminator_loss*a	   �k�>   �k�>      �?!   �k�>)@4�B��=2�ѩ�-�>���%�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	r�I���A�*�
w
discriminator_loss*a	    H��>    H��>      �?!    H��>)  "h�]�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        d��'�       �E��	�ݩI���A�*�
w
discriminator_loss*a	   ��.�>   ��.�>      �?!   ��.�>)@H�;E��=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        >Wz�       �E��	A"�I���A�*�
w
discriminator_loss*a	   @> �>   @> �>      �?!   @> �>)���y��=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        >
n�       �E��	�l�I���A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)����4\�=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        V�ݕ�       �E��	81�I���A�*�
w
discriminator_loss*a	   �h�>   �h�>      �?!   �h�>) ���y�=2['�?��>K+�E���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���d�       �E��	ր�I���A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>)��dM��=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        |]M��       �E��	FM�I���A�*�
w
discriminator_loss*a	    8G�>    8G�>      �?!    8G�>) pp3��>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �E$�       �E��	���L���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) d6%R�=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	S��L���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) iɖ���=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        pє��       �E��	���L���A�*�
w
discriminator_loss*a	   �=��>   �=��>      �?!   �=��>) ��2�V�=2
�/eq
�>;�"�q�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �y��       �E��	���L���A�*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)  P��=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �AZ��       �E��	��L���A�*�
w
discriminator_loss*a	   `��>   `��>      �?!   `��>)@�^��N�=2K+�E���>jqs&\��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ᤛx�       �E��	�3�L���A�*�
w
discriminator_loss*a	   `W��>   `W��>      �?!   `W��>) ��t\m�=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �(-��       �E��	p�M���A�*�
w
discriminator_loss*a	   �{Q�>   �{Q�>      �?!   �{Q�>) �S��=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��b�       �E��	i�
M���A�*�
w
discriminator_loss*a	   `�s�>   `�s�>      �?!   `�s�>)@�1V	�=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        9��m�       �E��	"DP���A�*�
w
discriminator_loss*a	   �4��>   �4��>      �?!   �4��>) �}C;�=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        h�`��       �E��	� P���A�*�
w
discriminator_loss*a	    DS�>    DS�>      �?!    DS�>)  �R�&�=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        '�H>�       �E��	��(P���A�*�
w
discriminator_loss*a	   @F��>   @F��>      �?!   @F��>)�8��{t�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �t��       �E��	g�0P���A�*�
w
discriminator_loss*a	    �~�>    �~�>      �?!    �~�>)  �=2�h���`�>�ߊ4F��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ;��!�       �E��	�i8P���A�*�
w
discriminator_loss*a	   �j��>   �j��>      �?!   �j��>)��W� >2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        T6>�       �E��	�?P���A�*�
w
discriminator_loss*a	    "��>    "��>      �?!    "��>)@�@�穵=2jqs&\��>��~]�[�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        v]�J�       �E��	,�FP���A�*�
w
discriminator_loss*a	   ��<�>   ��<�>      �?!   ��<�>) 7�َ.�=2a�Ϭ(�>8K�ߝ�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �,���       �E��	�|NP���A�*�
w
discriminator_loss*a	    �&�>    �&�>      �?!    �&�>) w�	>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        B�i�       �E��	��~S���A�*�
w
discriminator_loss*a	   �@+�>   �@+�>      �?!   �@+�>) 	�ơ�=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��@h�       �E��	v��S���A�*�
w
discriminator_loss*a	    �*�>    �*�>      �?!    �*�>) &4�Z>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�)�S���A�*�
w
discriminator_loss*a	   �;��>   �;��>      �?!   �;��>)��o�f��=2�iD*L��>E��a�W�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��LQ�       �E��	�S���A�*�
w
discriminator_loss*a	   �,��>   �,��>      �?!   �,��>) )��ۛ�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        i'���       �E��	�n�S���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)��$�1�	>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �YT_�       �E��	�S���A�*�
w
discriminator_loss*a	    �-�>    �-�>      �?!    �-�>) �t;s�=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �-g��       �E��	<�S���A�*�
w
discriminator_loss*a	   @� ?   @� ?      �?!   @� ?) �.�>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        T#���       �E��	��S���A�*�
w
discriminator_loss*a	    -	?    -	?      �?!    -	?) �7Q�#>2����?f�ʜ�7
?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        a����       �E��	���V���A�*�
w
discriminator_loss*a	    �f?    �f?      �?!    �f?) �R|�>2��[�?1��a˲?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	��V���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>) ĕL���=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��L��       �E��	�f�V���A�*�
w
discriminator_loss*a	    ]��>    ]��>      �?!    ]��>) ��DϞ�=2jqs&\��>��~]�[�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �!]e�       �E��	Q��V���A�*�
w
discriminator_loss*a	   @Ŀ�>   @Ŀ�>      �?!   @Ŀ�>) !��s��=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        m���       �E��	+O�V���A�*�
w
discriminator_loss*a	   @*��>   @*��>      �?!   @*��>) ��;���=2jqs&\��>��~]�[�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �U�       �E��	L��V���A�*�
w
discriminator_loss*a	   ���>   ���>      �?!   ���>)����>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �[^�       �E��	W���A�*�
w
discriminator_loss*a	    <��>    <��>      �?!    <��>) ��J��=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        I���       �E��	2fW���A�*�
w
discriminator_loss*a	   �/��>   �/��>      �?!   �/��>) �F�EB�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �~S��       �E��	�=Z���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �tkJ>2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���j�       �E��	�EZ���A�*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) Q�`&>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��<g�       �E��	$LZ���A�*�
w
discriminator_loss*a	   �!%?   �!%?      �?!   �!%?)@��#�[>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �yT��       M�	L�QZ���A*�
w
discriminator_loss*a	   `Ա�>   `Ա�>      �?!   `Ա�>) y�>` >2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �i���       ���g	j"~]���A*�
w
discriminator_loss*a	   �I��>   �I��>      �?!   �I��>)��%��9 >2I��P=�>��Zr[v�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        h�<��       ���g	ܠ�]���A
*�
w
discriminator_loss*a	    p�>    p�>      �?!    p�>)  �q���=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �p�       ���g	R��]���A*�
w
discriminator_loss*a	   `u��>   `u��>      �?!   `u��>) ��)S�=2E��a�W�>�ѩ�-�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #�z�       ���g	��]���A*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)�����>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �P�?�       ���g	�]���A*�
w
discriminator_loss*a	   �2��>   �2��>      �?!   �2��>)�|�����=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �t�l�       ���g	�(�]���A*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>)@�>',�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Z�%�       ���g	�P�]���A#*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@ܶ�P�=2�XQ��>�����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        G�A{�       ���g	�v�]���A(*�
w
discriminator_loss*a	    �3�>    �3�>      �?!    �3�>) �
��=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        y0�       ���g	��a���A-*�
w
discriminator_loss*a	    3��>    3��>      �?!    3��>) Hק\�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �<b�       ���g	0%a���A2*�
w
discriminator_loss*a	    6�>    6�>      �?!    6�>) ��dR	�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        A<��       ���g	c a���A7*�
w
discriminator_loss*a	   @��?   @��?      �?!   @��?) �X� >21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       ���g	ۍ'a���A<*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) �M�|�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ~K��       ���g	�.a���AA*�
w
discriminator_loss*a	   ����>   ����>      �?!   ����>) [��0��=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �bu��       ���g	��5a���AF*�
w
discriminator_loss*a	    !��>    !��>      �?!    !��>) ����=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��._�       ���g	{=a���AK*�
w
discriminator_loss*a	   ��b�>   ��b�>      �?!   ��b�>)@��S#}�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        p��}�       ���g	}Da���AP*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �7^�P�=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        J����       ���g	h�nd���AU*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) �W�(�=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        `����       ���g	�lwd���AZ*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) ��w�=2��(���>a�Ϭ(�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        D�#��       ���g	��~d���A_*�
w
discriminator_loss*a	    O��>    O��>      �?!    O��>) &���=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �@8�       ���g	���d���Ad*�
w
discriminator_loss*a	   @b��>   @b��>      �?!   @b��>) Q���=�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        gė�       ���g	�e�d���Ai*�
w
discriminator_loss*a	    �C�>    �C�>      �?!    �C�>)  $4���=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        7���       ���g	�j�d���An*�
w
discriminator_loss*a	   @9��>   @9��>      �?!   @9��>)�lf�>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �b�       ���g	�.�d���As*�
w
discriminator_loss*a	   �I��>   �I��>      �?!   �I��>) eف��=2E��a�W�>�ѩ�-�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �u���       ���g	ٖ�d���Ax*�
w
discriminator_loss*a	   @Y�>   @Y�>      �?!   @Y�>)���G�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        VA�c�       ���g	!4�g���A}*�
w
discriminator_loss*a	   �� ?   �� ?      �?!   �� ?) �k��=Q>2ji6�9�?�S�F !?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	ơh���A�*�
w
discriminator_loss*a	    p� ?    p� ?      �?!    p� ?)@��\��>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        <���       �E��	��
h���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) gټ �=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        &d��       �E��	A�h���A�*�
w
discriminator_loss*a	   @@��>   @@��>      �?!   @@��>)� ��ë�=2E��a�W�>�ѩ�-�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �l�       �E��	9&h���A�*�
w
discriminator_loss*a	   �\��>   �\��>      �?!   �\��>) b��c)>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        TJ�5�       �E��	B�"h���A�*�
w
discriminator_loss*a	   �<s�>   �<s�>      �?!   �<s�>) Ĕ�F�=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        |���       �E��	/�)h���A�*�
w
discriminator_loss*a	    ���>    ���>      �?!    ���>) z�Rf9>2��Zr[v�>O�ʗ��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�&1h���A�*�
w
discriminator_loss*a	   @ȯ�>   @ȯ�>      �?!   @ȯ�>) AB����=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Iv���       �E��	�\�k���A�*�
w
discriminator_loss*a	   �=��>   �=��>      �?!   �=��>)�(_߁u�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�Վk���A�*�
w
discriminator_loss*a	   @i��>   @i��>      �?!   @i��>)�,��W�=2E��a�W�>�ѩ�-�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �]���       �E��	S��k���A�*�
w
discriminator_loss*a	    %��>    %��>      �?!    %��>) �U��Q�=2��~]�[�>��>M|K�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��Q�       �E��	���k���A�*�
w
discriminator_loss*a	    B��>    B��>      �?!    B��>)  �5��=2E��a�W�>�ѩ�-�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �'��       �E��	w�k���A�*�
w
discriminator_loss*a	   @#��>   @#��>      �?!   @#��>) ��Iı=2K+�E���>jqs&\��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        qz�n�       �E��	��k���A�*�
w
discriminator_loss*a	    ,��>    ,��>      �?!    ,��>) ��Lj��=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        Yb0��       �E��	��k���A�*�
w
discriminator_loss*a	   ��.�>   ��.�>      �?!   ��.�>) ı��t�=2�uE����>�f����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �ii�       �E��	*��k���A�*�
w
discriminator_loss*a	   �/��>   �/��>      �?!   �/��>) �ޝ�"�=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �6��       �E��	9�bo���A�*�
w
discriminator_loss*a	   �&��>   �&��>      �?!   �&��>) �u���=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�jo���A�*�
w
discriminator_loss*a	    �}?    �}?      �?!    �}?) �>O�->2>h�'�?x?�x�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        B��H�       �E��	�(ro���A�*�
w
discriminator_loss*a	   @�� ?   @�� ?      �?!   @�� ?) 9`�%�>2�FF�G ?��[�?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        v��V�       �E��	;dyo���A�*�
w
discriminator_loss*a	    >�>    >�>      �?!    >�>) 0|5�=2pz�w�7�>I��P=�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        0�d�       �E��	fπo���A�*�
w
discriminator_loss*a	   @�E�>   @�E�>      �?!   @�E�>) A� ^H�=2�f����>��(���>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	���o���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) ����-�=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �H�\�       �E��	��o���A�*�
w
discriminator_loss*a	    �g�>    �g�>      �?!    �g�>) 
G��w>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        a۷��       �E��	Wv�o���A�*�
w
discriminator_loss*a	    ,��>    ,��>      �?!    ,��>) �<v�p	>2O�ʗ��>>�?�s��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ǲ���       �E��	)8s���A�*�
w
discriminator_loss*a	   �H*�>   �H*�>      �?!   �H*�>)�d���?�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        `}���       �E��	�\@s���A�*�
w
discriminator_loss*a	   @�d�>   @�d�>      �?!   @�d�>)�쭘��>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��;�       �E��	�Hs���A�*�
w
discriminator_loss*a	   `H(?   `H(?      �?!   `H(?)@b'[e>21��a˲?6�]��?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        %��1�       �E��	�SPs���A�*�
w
discriminator_loss*a	   �ڐ�>   �ڐ�>      �?!   �ڐ�>) �.E�H�=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �1_t�       �E��	z�Ws���A�*�
w
discriminator_loss*a	    0 ?    0 ?      �?!    0 ?)@�ֲ�`>2>�?�s��>�FF�G ?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �{t�       �E��	
_s���A�*�
w
discriminator_loss*a	   @2�>   @2�>      �?!   @2�>)��T��=28K�ߝ�>�h���`�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��ct�       �E��	M*js���A�*�
w
discriminator_loss*a	   �D��>   �D��>      �?!   �D��>) D%��D�=2��~]�[�>��>M|K�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        #߷��       �E��	@	ss���A�*�
w
discriminator_loss*a	   �:w%?   �:w%?      �?!   �:w%?) ���r�\>2U�4@@�$?+A�F�&?�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        c���       �E��	�'w���A�*�
w
discriminator_loss*a	   @l��>   @l��>      �?!   @l��>)�0�0kg�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	�D/w���A�*�
w
discriminator_loss*a	   @���>   @���>      �?!   @���>) ٚRؙ�=2})�l a�>pz�w�7�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ���       �E��	��6w���A�*�
w
discriminator_loss*a	   �a��>   �a��>      �?!   �a��>) 1��
�=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �E(��       �E��	��=w���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>)@X��m(�=2�ߊ4F��>})�l a�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��q�       �E��	�)Ew���A�*�
w
discriminator_loss*a	   @��>   @��>      �?!   @��>) Y�~�=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        vn���       �E��	׉Lw���A�*�
w
discriminator_loss*a	   �[^�>   �[^�>      �?!   �[^�>) YR�Ph�=2�iD*L��>E��a�W�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ����       �E��	�Tw���A�*�
w
discriminator_loss*a	   �e��>   �e��>      �?!   �e��>) T�qLX�=2�_�T�l�>�iD*L��>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �<Ua�       �E��	h�[w���A�*�
w
discriminator_loss*a	    q��>    q��>      �?!    q��>) �OM��=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	ܘ�z���A�*�
w
discriminator_loss*a	    ��>    ��>      �?!    ��>) @���:�=2��>M|K�>�_�T�l�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        ��|��       �E��	=��z���A�*�
w
discriminator_loss*a	    5��>    5��>      �?!    5��>)@d4j�0�=2���%�>�uE����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �����       �E��	���z���A�*�
w
discriminator_loss*a	   �p�>   �p�>      �?!   �p�>)@��@0�=2�XQ��>�����>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        n�i��       �E��	��z���A�*�
w
discriminator_loss*a	   �H�>   �H�>      �?!   �H�>)����l�=2
�/eq
�>;�"�q�>�������:              �?        
O
generator_loss*=      �?2        �-���q=�������:              �?        �h��       �E��	��z���A�*�
w
discri