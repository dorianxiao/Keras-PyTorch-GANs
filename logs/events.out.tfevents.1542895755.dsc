       ŁK"	  Ŕ"ŽýÖAbrain.Event:2uÝŮ|ä     ˇňy	~+ü"ŽýÖA"ďČ
r
Generator/noisePlaceholder*
shape:˙˙˙˙˙˙˙˙˙d*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
ń
VGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
ă
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&ž*
dtype0*
_output_shapes
: 
ă
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&>
á
^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	d*

seed*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
seed2
ň
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel

TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/sub*
_output_shapes
:	d*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
÷
PGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniformAddTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d*
T0
ő
5Generator/first/Generator/firstfully_connected/kernel
VariableV2*
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container 
ě
<Generator/first/Generator/firstfully_connected/kernel/AssignAssign5Generator/first/Generator/firstfully_connected/kernelPGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d
ń
:Generator/first/Generator/firstfully_connected/kernel/readIdentity5Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
Ü
EGenerator/first/Generator/firstfully_connected/bias/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
é
3Generator/first/Generator/firstfully_connected/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
×
:Generator/first/Generator/firstfully_connected/bias/AssignAssign3Generator/first/Generator/firstfully_connected/biasEGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:
ç
8Generator/first/Generator/firstfully_connected/bias/readIdentity3Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
ĺ
5Generator/first/Generator/firstfully_connected/MatMulMatMulGenerator/noise:Generator/first/Generator/firstfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
ü
6Generator/first/Generator/firstfully_connected/BiasAddBiasAdd5Generator/first/Generator/firstfully_connected/MatMul8Generator/first/Generator/firstfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
/Generator/first/Generator/firstleaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Đ
-Generator/first/Generator/firstleaky_relu/mulMul/Generator/first/Generator/firstleaky_relu/alpha6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
)Generator/first/Generator/firstleaky_reluMaximum-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
XGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"      *
dtype0
ç
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/minConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   ž*
dtype0*
_output_shapes
: 
ç
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
č
`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformXGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
seed2
ú
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/subSubVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
_output_shapes
: 

VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulMul`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/sub*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:


RGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniformAddVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel
ű
7Generator/second/Generator/secondfully_connected/kernel
VariableV2*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ő
>Generator/second/Generator/secondfully_connected/kernel/AssignAssign7Generator/second/Generator/secondfully_connected/kernelRGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:

ř
<Generator/second/Generator/secondfully_connected/kernel/readIdentity7Generator/second/Generator/secondfully_connected/kernel*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:

ŕ
GGenerator/second/Generator/secondfully_connected/bias/Initializer/zerosConst*
_output_shapes	
:*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB*    *
dtype0
í
5Generator/second/Generator/secondfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:
ß
<Generator/second/Generator/secondfully_connected/bias/AssignAssign5Generator/second/Generator/secondfully_connected/biasGGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:
í
:Generator/second/Generator/secondfully_connected/bias/readIdentity5Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias

7Generator/second/Generator/secondfully_connected/MatMulMatMul)Generator/first/Generator/firstleaky_relu<Generator/second/Generator/secondfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

8Generator/second/Generator/secondfully_connected/BiasAddBiasAdd7Generator/second/Generator/secondfully_connected/MatMul:Generator/second/Generator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ă
HGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/onesConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ń
7Generator/second/Generator/secondbatch_normalized/gamma
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma
ć
>Generator/second/Generator/secondbatch_normalized/gamma/AssignAssign7Generator/second/Generator/secondbatch_normalized/gammaHGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ó
<Generator/second/Generator/secondbatch_normalized/gamma/readIdentity7Generator/second/Generator/secondbatch_normalized/gamma*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:*
T0
â
HGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB*    *
dtype0*
_output_shapes	
:
ď
6Generator/second/Generator/secondbatch_normalized/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:
ă
=Generator/second/Generator/secondbatch_normalized/beta/AssignAssign6Generator/second/Generator/secondbatch_normalized/betaHGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
đ
;Generator/second/Generator/secondbatch_normalized/beta/readIdentity6Generator/second/Generator/secondbatch_normalized/beta*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:
đ
OGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
valueB*    
ý
=Generator/second/Generator/secondbatch_normalized/moving_mean
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean
˙
DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignAssign=Generator/second/Generator/secondbatch_normalized/moving_meanOGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:

BGenerator/second/Generator/secondbatch_normalized/moving_mean/readIdentity=Generator/second/Generator/secondbatch_normalized/moving_mean*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
_output_shapes	
:
÷
RGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/onesConst*
_output_shapes	
:*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
valueB*  ?*
dtype0

AGenerator/second/Generator/secondbatch_normalized/moving_variance
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
	container *
shape:

HGenerator/second/Generator/secondbatch_normalized/moving_variance/AssignAssignAGenerator/second/Generator/secondbatch_normalized/moving_varianceRGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/ones*
use_locking(*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:

FGenerator/second/Generator/secondbatch_normalized/moving_variance/readIdentityAGenerator/second/Generator/secondbatch_normalized/moving_variance*
_output_shapes	
:*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance

AGenerator/second/Generator/secondbatch_normalized/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o:*
dtype0
÷
?Generator/second/Generator/secondbatch_normalized/batchnorm/addAddFGenerator/second/Generator/secondbatch_normalized/moving_variance/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/add/y*
_output_shapes	
:*
T0
ą
AGenerator/second/Generator/secondbatch_normalized/batchnorm/RsqrtRsqrt?Generator/second/Generator/secondbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:
í
?Generator/second/Generator/secondbatch_normalized/batchnorm/mulMulAGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt<Generator/second/Generator/secondbatch_normalized/gamma/read*
T0*
_output_shapes	
:
ö
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAdd?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2MulBGenerator/second/Generator/secondbatch_normalized/moving_mean/read?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:
ě
?Generator/second/Generator/secondbatch_normalized/batchnorm/subSub;Generator/second/Generator/secondbatch_normalized/beta/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2*
_output_shapes	
:*
T0
˙
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1AddAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1?Generator/second/Generator/secondbatch_normalized/batchnorm/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
1Generator/second/Generator/secondleaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL>
ß
/Generator/second/Generator/secondleaky_relu/mulMul1Generator/second/Generator/secondleaky_relu/alphaAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
+Generator/second/Generator/secondleaky_reluMaximum/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
VGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ă
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *óľ˝*
dtype0
ă
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *óľ=
â
^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
seed2B
ň
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel

TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:

ř
PGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniformAddTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:

÷
5Generator/third/Generator/thirdfully_connected/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
í
<Generator/third/Generator/thirdfully_connected/kernel/AssignAssign5Generator/third/Generator/thirdfully_connected/kernelPGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:

ň
:Generator/third/Generator/thirdfully_connected/kernel/readIdentity5Generator/third/Generator/thirdfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:

Ü
EGenerator/third/Generator/thirdfully_connected/bias/Initializer/zerosConst*
_output_shapes	
:*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB*    *
dtype0
é
3Generator/third/Generator/thirdfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:
×
:Generator/third/Generator/thirdfully_connected/bias/AssignAssign3Generator/third/Generator/thirdfully_connected/biasEGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:
ç
8Generator/third/Generator/thirdfully_connected/bias/readIdentity3Generator/third/Generator/thirdfully_connected/bias*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:

5Generator/third/Generator/thirdfully_connected/MatMulMatMul+Generator/second/Generator/secondleaky_relu:Generator/third/Generator/thirdfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ü
6Generator/third/Generator/thirdfully_connected/BiasAddBiasAdd5Generator/third/Generator/thirdfully_connected/MatMul8Generator/third/Generator/thirdfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
FGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/onesConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
í
5Generator/third/Generator/thirdbatch_normalized/gamma
VariableV2*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
Ţ
<Generator/third/Generator/thirdbatch_normalized/gamma/AssignAssign5Generator/third/Generator/thirdbatch_normalized/gammaFGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
í
:Generator/third/Generator/thirdbatch_normalized/gamma/readIdentity5Generator/third/Generator/thirdbatch_normalized/gamma*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:*
T0
Ţ
FGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zerosConst*
_output_shapes	
:*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB*    *
dtype0
ë
4Generator/third/Generator/thirdbatch_normalized/beta
VariableV2*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
Ű
;Generator/third/Generator/thirdbatch_normalized/beta/AssignAssign4Generator/third/Generator/thirdbatch_normalized/betaFGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ę
9Generator/third/Generator/thirdbatch_normalized/beta/readIdentity4Generator/third/Generator/thirdbatch_normalized/beta*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:
ě
MGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zerosConst*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ů
;Generator/third/Generator/thirdbatch_normalized/moving_mean
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean
÷
BGenerator/third/Generator/thirdbatch_normalized/moving_mean/AssignAssign;Generator/third/Generator/thirdbatch_normalized/moving_meanMGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:
˙
@Generator/third/Generator/thirdbatch_normalized/moving_mean/readIdentity;Generator/third/Generator/thirdbatch_normalized/moving_mean*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
_output_shapes	
:
ó
PGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/onesConst*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:

?Generator/third/Generator/thirdbatch_normalized/moving_variance
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
	container 

FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignAssign?Generator/third/Generator/thirdbatch_normalized/moving_variancePGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(

DGenerator/third/Generator/thirdbatch_normalized/moving_variance/readIdentity?Generator/third/Generator/thirdbatch_normalized/moving_variance*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
_output_shapes	
:

?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o:*
dtype0
ń
=Generator/third/Generator/thirdbatch_normalized/batchnorm/addAddDGenerator/third/Generator/thirdbatch_normalized/moving_variance/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/y*
_output_shapes	
:*
T0
­
?Generator/third/Generator/thirdbatch_normalized/batchnorm/RsqrtRsqrt=Generator/third/Generator/thirdbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:
ç
=Generator/third/Generator/thirdbatch_normalized/batchnorm/mulMul?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:
đ
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAdd=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2Mul@Generator/third/Generator/thirdbatch_normalized/moving_mean/read=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:
ć
=Generator/third/Generator/thirdbatch_normalized/batchnorm/subSub9Generator/third/Generator/thirdbatch_normalized/beta/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:
ů
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1Add?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
/Generator/third/Generator/thirdleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *ÍĚL>*
dtype0
Ů
-Generator/third/Generator/thirdleaky_relu/mulMul/Generator/third/Generator/thirdleaky_relu/alpha?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
×
)Generator/third/Generator/thirdleaky_reluMaximum-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
VGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      
ă
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  ˝*
dtype0*
_output_shapes
: 
ă
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
â
^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
seed2m
ň
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
_output_shapes
: 

TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
ř
PGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniformAddTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:

÷
5Generator/forth/Generator/forthfully_connected/kernel
VariableV2* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container *
shape:
*
dtype0
í
<Generator/forth/Generator/forthfully_connected/kernel/AssignAssign5Generator/forth/Generator/forthfully_connected/kernelPGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(
ň
:Generator/forth/Generator/forthfully_connected/kernel/readIdentity5Generator/forth/Generator/forthfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:

č
UGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
Ř
KGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
ń
EGenerator/forth/Generator/forthfully_connected/bias/Initializer/zerosFillUGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorKGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/Const*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0
é
3Generator/forth/Generator/forthfully_connected/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container 
×
:Generator/forth/Generator/forthfully_connected/bias/AssignAssign3Generator/forth/Generator/forthfully_connected/biasEGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:
ç
8Generator/forth/Generator/forthfully_connected/bias/readIdentity3Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
˙
5Generator/forth/Generator/forthfully_connected/MatMulMatMul)Generator/third/Generator/thirdleaky_relu:Generator/forth/Generator/forthfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ü
6Generator/forth/Generator/forthfully_connected/BiasAddBiasAdd5Generator/forth/Generator/forthfully_connected/MatMul8Generator/forth/Generator/forthfully_connected/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
ë
VGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:*
dtype0*
_output_shapes
:
Ű
LGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *  ?*
dtype0
ö
FGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/onesFillVGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:
í
5Generator/forth/Generator/forthbatch_normalized/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:
Ţ
<Generator/forth/Generator/forthbatch_normalized/gamma/AssignAssign5Generator/forth/Generator/forthbatch_normalized/gammaFGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
í
:Generator/forth/Generator/forthbatch_normalized/gamma/readIdentity5Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
ę
VGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:*
dtype0*
_output_shapes
:
Ú
LGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ő
FGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zerosFillVGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:
ë
4Generator/forth/Generator/forthbatch_normalized/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:
Ű
;Generator/forth/Generator/forthbatch_normalized/beta/AssignAssign4Generator/forth/Generator/forthbatch_normalized/betaFGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ę
9Generator/forth/Generator/forthbatch_normalized/beta/readIdentity4Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
ř
]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB:
č
SGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/ConstConst*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 

MGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/Const*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*

index_type0*
_output_shapes	
:
ů
;Generator/forth/Generator/forthbatch_normalized/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
	container 
÷
BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignAssign;Generator/forth/Generator/forthbatch_normalized/moving_meanMGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
˙
@Generator/forth/Generator/forthbatch_normalized/moving_mean/readIdentity;Generator/forth/Generator/forthbatch_normalized/moving_mean*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
_output_shapes	
:*
T0
˙
`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorConst*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB:*
dtype0*
_output_shapes
:
ď
VGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/ConstConst*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 

PGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/onesFill`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorVGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/Const*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*

index_type0*
_output_shapes	
:*
T0

?Generator/forth/Generator/forthbatch_normalized/moving_variance
VariableV2*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignAssign?Generator/forth/Generator/forthbatch_normalized/moving_variancePGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

DGenerator/forth/Generator/forthbatch_normalized/moving_variance/readIdentity?Generator/forth/Generator/forthbatch_normalized/moving_variance*
_output_shapes	
:*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance

?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
ń
=Generator/forth/Generator/forthbatch_normalized/batchnorm/addAddDGenerator/forth/Generator/forthbatch_normalized/moving_variance/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:
­
?Generator/forth/Generator/forthbatch_normalized/batchnorm/RsqrtRsqrt=Generator/forth/Generator/forthbatch_normalized/batchnorm/add*
_output_shapes	
:*
T0
ç
=Generator/forth/Generator/forthbatch_normalized/batchnorm/mulMul?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt:Generator/forth/Generator/forthbatch_normalized/gamma/read*
T0*
_output_shapes	
:
đ
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAdd=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2Mul@Generator/forth/Generator/forthbatch_normalized/moving_mean/read=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:
ć
=Generator/forth/Generator/forthbatch_normalized/batchnorm/subSub9Generator/forth/Generator/forthbatch_normalized/beta/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:
ů
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1Add?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
/Generator/forth/Generator/forthleaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Ů
-Generator/forth/Generator/forthleaky_relu/mulMul/Generator/forth/Generator/forthleaky_relu/alpha?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
)Generator/forth/Generator/forthleaky_reluMaximum-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
7Generator/dense/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ľ
5Generator/dense/kernel/Initializer/random_uniform/minConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *zők˝*
dtype0*
_output_shapes
: 
Ľ
5Generator/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *)
_class
loc:@Generator/dense/kernel*
valueB
 *zők=*
dtype0

?Generator/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7Generator/dense/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed*
T0*)
_class
loc:@Generator/dense/kernel*
seed2˘
ö
5Generator/dense/kernel/Initializer/random_uniform/subSub5Generator/dense/kernel/Initializer/random_uniform/max5Generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@Generator/dense/kernel*
_output_shapes
: 

5Generator/dense/kernel/Initializer/random_uniform/mulMul?Generator/dense/kernel/Initializer/random_uniform/RandomUniform5Generator/dense/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*)
_class
loc:@Generator/dense/kernel
ü
1Generator/dense/kernel/Initializer/random_uniformAdd5Generator/dense/kernel/Initializer/random_uniform/mul5Generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:

š
Generator/dense/kernel
VariableV2*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ń
Generator/dense/kernel/AssignAssignGenerator/dense/kernel1Generator/dense/kernel/Initializer/random_uniform*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

Generator/dense/kernel/readIdentityGenerator/dense/kernel*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:


&Generator/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*'
_class
loc:@Generator/dense/bias*
valueB*    
Ť
Generator/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@Generator/dense/bias
Ű
Generator/dense/bias/AssignAssignGenerator/dense/bias&Generator/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:

Generator/dense/bias/readIdentityGenerator/dense/bias*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:*
T0
Á
Generator/dense/MatMulMatMul)Generator/forth/Generator/forthleaky_reluGenerator/dense/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

Generator/dense/BiasAddBiasAddGenerator/dense/MatMulGenerator/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
Generator/TanhTanhGenerator/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
Discriminator/realPlaceholder*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙

^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ó
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/minConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HY˝*
dtype0*
_output_shapes
: 
ó
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HY=
ű
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shape*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
seed2´*
dtype0* 
_output_shapes
:
*

seed

\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/subSub\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/max\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
Ś
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mulMulfDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniform\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/sub*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:


XDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniformAdd\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mul\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:
*
T0

=Discriminator/first/Discriminator/firstfully_connected/kernel
VariableV2*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 

DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignAssign=Discriminator/first/Discriminator/firstfully_connected/kernelXDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:


BDiscriminator/first/Discriminator/firstfully_connected/kernel/readIdentity=Discriminator/first/Discriminator/firstfully_connected/kernel*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:

ě
MDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ů
;Discriminator/first/Discriminator/firstfully_connected/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
÷
BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignAssign;Discriminator/first/Discriminator/firstfully_connected/biasMDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:
˙
@Discriminator/first/Discriminator/firstfully_connected/bias/readIdentity;Discriminator/first/Discriminator/firstfully_connected/bias*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:
ř
=Discriminator/first/Discriminator/firstfully_connected/MatMulMatMulDiscriminator/realBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

>Discriminator/first/Discriminator/firstfully_connected/BiasAddBiasAdd=Discriminator/first/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
7Discriminator/first/Discriminator/firstleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *ÍĚL>*
dtype0
č
5Discriminator/first/Discriminator/firstleaky_relu/mulMul7Discriminator/first/Discriminator/firstleaky_relu/alpha>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ć
1Discriminator/first/Discriminator/firstleaky_reluMaximum5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
÷
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *óľ˝
÷
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *óľ=*
dtype0*
_output_shapes
: 

hDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
seed2Ç

^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/subSub^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/max^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
Ž
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mulMulhDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniform^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
 
ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniformAdd^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mul^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:


?Discriminator/second/Discriminator/secondfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:


FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignAssign?Discriminator/second/Discriminator/secondfully_connected/kernelZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(

DDiscriminator/second/Discriminator/secondfully_connected/kernel/readIdentity?Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
đ
ODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB*    
ý
=Discriminator/second/Discriminator/secondfully_connected/bias
VariableV2*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
˙
DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignAssign=Discriminator/second/Discriminator/secondfully_connected/biasODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:

BDiscriminator/second/Discriminator/secondfully_connected/bias/readIdentity=Discriminator/second/Discriminator/secondfully_connected/bias*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:

?Discriminator/second/Discriminator/secondfully_connected/MatMulMatMul1Discriminator/first/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

@Discriminator/second/Discriminator/secondfully_connected/BiasAddBiasAdd?Discriminator/second/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
~
9Discriminator/second/Discriminator/secondleaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
î
7Discriminator/second/Discriminator/secondleaky_relu/mulMul9Discriminator/second/Discriminator/secondleaky_relu/alpha@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ě
3Discriminator/second/Discriminator/secondleaky_reluMaximum7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
9Discriminator/out/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@Discriminator/out/kernel*
valueB"      *
dtype0*
_output_shapes
:
Š
7Discriminator/out/kernel/Initializer/random_uniform/minConst*+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Ivž*
dtype0*
_output_shapes
: 
Š
7Discriminator/out/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 

ADiscriminator/out/kernel/Initializer/random_uniform/RandomUniformRandomUniform9Discriminator/out/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed*
T0*+
_class!
loc:@Discriminator/out/kernel*
seed2Ú
ţ
7Discriminator/out/kernel/Initializer/random_uniform/subSub7Discriminator/out/kernel/Initializer/random_uniform/max7Discriminator/out/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
: 

7Discriminator/out/kernel/Initializer/random_uniform/mulMulADiscriminator/out/kernel/Initializer/random_uniform/RandomUniform7Discriminator/out/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*+
_class!
loc:@Discriminator/out/kernel

3Discriminator/out/kernel/Initializer/random_uniformAdd7Discriminator/out/kernel/Initializer/random_uniform/mul7Discriminator/out/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	
ť
Discriminator/out/kernel
VariableV2*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
ř
Discriminator/out/kernel/AssignAssignDiscriminator/out/kernel3Discriminator/out/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel

Discriminator/out/kernel/readIdentityDiscriminator/out/kernel*
_output_shapes
:	*
T0*+
_class!
loc:@Discriminator/out/kernel
 
(Discriminator/out/bias/Initializer/zerosConst*
_output_shapes
:*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0
­
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
â
Discriminator/out/bias/AssignAssignDiscriminator/out/bias(Discriminator/out/bias/Initializer/zeros*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

Discriminator/out/bias/readIdentityDiscriminator/out/bias*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
Î
Discriminator/out/MatMulMatMul3Discriminator/second/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
¤
Discriminator/out/BiasAddBiasAddDiscriminator/out/MatMulDiscriminator/out/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
q
Discriminator/out/SigmoidSigmoidDiscriminator/out/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ö
?Discriminator/first_1/Discriminator/firstfully_connected/MatMulMatMulGenerator/TanhBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

@Discriminator/first_1/Discriminator/firstfully_connected/BiasAddBiasAdd?Discriminator/first_1/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
~
9Discriminator/first_1/Discriminator/firstleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *ÍĚL>*
dtype0
î
7Discriminator/first_1/Discriminator/firstleaky_relu/mulMul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
3Discriminator/first_1/Discriminator/firstleaky_reluMaximum7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

ADiscriminator/second_1/Discriminator/secondfully_connected/MatMulMatMul3Discriminator/first_1/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

BDiscriminator/second_1/Discriminator/secondfully_connected/BiasAddBiasAddADiscriminator/second_1/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC

;Discriminator/second_1/Discriminator/secondleaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
ô
9Discriminator/second_1/Discriminator/secondleaky_relu/mulMul;Discriminator/second_1/Discriminator/secondleaky_relu/alphaBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ň
5Discriminator/second_1/Discriminator/secondleaky_reluMaximum9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
Discriminator/out_1/MatMulMatMul5Discriminator/second_1/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
¨
Discriminator/out_1/BiasAddBiasAddDiscriminator/out_1/MatMulDiscriminator/out/bias/read*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
u
Discriminator/out_1/SigmoidSigmoidDiscriminator/out_1/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
W
LogLogDiscriminator/out/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
`
subSubsub/xDiscriminator/out_1/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
C
Log_1Logsub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
H
addAddLogLog_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
A
NegNegadd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
discriminator_loss/tagConst*
_output_shapes
: *#
valueB Bdiscriminator_loss*
dtype0
d
discriminator_lossHistogramSummarydiscriminator_loss/tagNeg*
_output_shapes
: *
T0
L
sub_1/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
sub_1Subsub_1/xDiscriminator/out_1/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
E
Log_2Logsub_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
generator_loss/tagConst*
valueB Bgenerator_loss*
dtype0*
_output_shapes
: 
^
generator_lossHistogramSummarygenerator_loss/tagLog_2*
T0*
_output_shapes
: 
R
gradients/ShapeShapeNeg*
T0*
out_type0*
_output_shapes
:
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

gradients/FillFillgradients/Shapegradients/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
_
gradients/Neg_grad/NegNeggradients/Fill*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
[
gradients/add_grad/ShapeShapeLog*
T0*
out_type0*
_output_shapes
:
_
gradients/add_grad/Shape_1ShapeLog_1*
_output_shapes
:*
T0*
out_type0
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/add_grad/SumSumgradients/Neg_grad/Neg(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ł
gradients/add_grad/Sum_1Sumgradients/Neg_grad/Neg*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ú
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
gradients/Log_grad/Reciprocal
ReciprocalDiscriminator/out/Sigmoid,^gradients/add_grad/tuple/control_dependency*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Log_grad/mulMul+gradients/add_grad/tuple/control_dependencygradients/Log_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Log_1_grad/Reciprocal
Reciprocalsub.^gradients/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
gradients/Log_1_grad/mulMul-gradients/add_grad/tuple/control_dependency_1gradients/Log_1_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
4gradients/Discriminator/out/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out/Sigmoidgradients/Log_grad/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ą
gradients/sub_grad/SumSumgradients/Log_1_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ľ
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

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
É
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
: 
ŕ
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
ľ
4gradients/Discriminator/out/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*
data_formatNHWC*
_output_shapes
:*
T0
Ż
9gradients/Discriminator/out/BiasAdd_grad/tuple/group_depsNoOp5^gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad5^gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad
ş
Agradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyIdentity4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
Cgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad
Ă
6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid-gradients/sub_grad/tuple/control_dependency_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
.gradients/Discriminator/out/MatMul_grad/MatMulMatMulAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0

0gradients/Discriminator/out/MatMul_grad/MatMul_1MatMul3Discriminator/second/Discriminator/secondleaky_reluAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
¤
8gradients/Discriminator/out/MatMul_grad/tuple/group_depsNoOp/^gradients/Discriminator/out/MatMul_grad/MatMul1^gradients/Discriminator/out/MatMul_grad/MatMul_1
­
@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyIdentity.gradients/Discriminator/out/MatMul_grad/MatMul9^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients/Discriminator/out/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ş
Bgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Identity0gradients/Discriminator/out/MatMul_grad/MatMul_19^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
_output_shapes
:	
š
6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
ľ
;gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad7^gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
Â
Cgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad
ż
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeShape7Discriminator/second/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ę
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ę
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Shape@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
ą
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zerosFillJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0

Ogradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Xgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ă
Igradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectSelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqual@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
Kgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1SelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Fgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumSumIgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectXgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
¨
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1SumKgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1Zgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ž
Lgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
÷
Sgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeM^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1

[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeT^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*]
_classS
QOloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape
Ą
]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1T^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
÷
0gradients/Discriminator/out_1/MatMul_grad/MatMulMatMulCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

2gradients/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
Ş
:gradients/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp1^gradients/Discriminator/out_1/MatMul_grad/MatMul3^gradients/Discriminator/out_1/MatMul_grad/MatMul_1
ľ
Bgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity0gradients/Discriminator/out_1/MatMul_grad/MatMul;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity2gradients/Discriminator/out_1/MatMul_grad/MatMul_1;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	

gradients/AddNAddNCgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:

Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Î
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Đ
\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ł
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/MulMul[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumSumJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˘
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0

Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul9Discriminator/second/Discriminator/secondleaky_relu/alpha[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ş
Pgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Wgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1

_gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
ą
agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Î
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
Î
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
ˇ
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Qgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
Zgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ë
Kgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
Mgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
Hgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumKgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectZgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ž
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeHgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ž
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumMgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1\gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
´
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ý
Ugradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpM^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeO^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
Ł
]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeV^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1V^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/AddN_1AddNBgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
N*
_output_shapes
:	
÷
gradients/AddN_2AddN]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:
Ů
`gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_2\^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
ý
hgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_2a^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ě
jgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ň
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ö
^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapePgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Š
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
¨
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0
¤
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1`gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ŕ
Rgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Ygradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeS^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
Ą
agradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeZ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
š
cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1Z^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
Ugradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ď
Wgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul1Discriminator/first/Discriminator/firstleaky_reluhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

_gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
É
ggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
igradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

ý
gradients/AddN_3AddN_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
T0*
data_formatNHWC*
_output_shapes	
:
Ý
bgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_3^^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad

jgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3c^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradc^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ť
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeShape5Discriminator/first/Discriminator/firstleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ć
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
ď
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Shapeggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ť
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosFillHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Mgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
Vgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Ggradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectSelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Igradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1SelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
Dgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumSumGgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectVgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
˘
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeReshapeDgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
˛
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1SumIgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1Xgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
¨
Jgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
Qgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpI^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeK^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1

Ygradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeR^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1R^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
Wgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMuljgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ő
Ygradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relujgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

agradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulZ^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
Ń
igradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulb^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*j
_class`
^\loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ď
kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1b^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*l
_classb
`^loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1

gradients/AddN_4AddNjgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:

Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Ę
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ę
Zgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulMulYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumSumHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulZgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul7Discriminator/first/Discriminator/firstleaky_relu/alphaYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1\gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
´
Ngradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ý
Ugradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpM^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeO^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1

]gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeV^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*_
_classU
SQloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
Š
_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1V^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ę
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
ó
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapeigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ą
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ogradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ä
Xgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Igradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Kgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
Fgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumIgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectXgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
¨
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumKgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1Zgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ž
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
÷
Sgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeM^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1

[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeT^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1T^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/AddN_5AddNigradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:

ń
gradients/AddN_6AddN[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ˇ
Ygradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*
data_formatNHWC*
_output_shapes	
:
Ő
^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6Z^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
÷
fgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
hgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad

Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Î
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Đ
\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ł
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˘
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ş
Pgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Wgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1

_gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: 
ą
agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*c
_classY
WUloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
â
Sgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ź
Ugradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/realfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

]gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpT^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulV^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
Á
egradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentitySgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
ggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityUgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
÷
gradients/AddN_7AddN]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
T0*
data_formatNHWC*
_output_shapes	
:
Ů
`gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_7\^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
ý
hgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_7a^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ć
Ugradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ź
Wgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

_gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
É
ggradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*h
_class^
\Zloc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul
Ç
igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*j
_class`
^\loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1

gradients/AddN_8AddNhgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:*
T0

gradients/AddN_9AddNggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
*
T0
Ž
beta1_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
ż
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
Ţ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 

beta1_power/readIdentitybeta1_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
Ž
beta2_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *wž?*
dtype0*
_output_shapes
: 
ż
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
Ţ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(

beta2_power/readIdentitybeta2_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 

dDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ń
ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
­
TDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zerosFilldDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0* 
_output_shapes
:


BDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel

IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamTDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:


GDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:


fDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0
ó
\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ł
VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zerosFillfDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0

DDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1
VariableV2*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:


KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignAssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:


IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/readIdentityDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1* 
_output_shapes
:
*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
ń
RDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ţ
@Discriminator/first/Discriminator/firstfully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:

GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignAssign@Discriminator/first/Discriminator/firstfully_connected/bias/AdamRDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias

EDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/readIdentity@Discriminator/first/Discriminator/firstfully_connected/bias/Adam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:
ó
TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:

BDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:

IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:*
T0

fDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      
ő
\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ľ
VDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zerosFillfDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensor\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0

DDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:


KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamVDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:


IDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:


hDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
÷
^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    *
dtype0
ť
XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zerosFillhDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:


FDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1
VariableV2* 
_output_shapes
:
*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
*
dtype0
Ą
MDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/AssignAssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/readIdentityFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:

ő
TDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:

BDiscriminator/second/Discriminator/secondfully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:

IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/AssignAssignBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamTDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias

GDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/readIdentityBDiscriminator/second/Discriminator/secondfully_connected/bias/Adam*
_output_shapes	
:*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias
÷
VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:

DDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1
VariableV2*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:

KDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(

IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:
ł
/Discriminator/out/kernel/Adam/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Ŕ
Discriminator/out/kernel/Adam
VariableV2*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
ţ
$Discriminator/out/kernel/Adam/AssignAssignDiscriminator/out/kernel/Adam/Discriminator/out/kernel/Adam/Initializer/zeros*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
¤
"Discriminator/out/kernel/Adam/readIdentityDiscriminator/out/kernel/Adam*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	
ľ
1Discriminator/out/kernel/Adam_1/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Â
Discriminator/out/kernel/Adam_1
VariableV2*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	

&Discriminator/out/kernel/Adam_1/AssignAssignDiscriminator/out/kernel/Adam_11Discriminator/out/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel
¨
$Discriminator/out/kernel/Adam_1/readIdentityDiscriminator/out/kernel/Adam_1*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	
Ľ
-Discriminator/out/bias/Adam/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
˛
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
ń
"Discriminator/out/bias/Adam/AssignAssignDiscriminator/out/bias/Adam-Discriminator/out/bias/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:

 Discriminator/out/bias/Adam/readIdentityDiscriminator/out/bias/Adam*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:*
T0
§
/Discriminator/out/bias/Adam_1/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
´
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
÷
$Discriminator/out/bias/Adam_1/AssignAssignDiscriminator/out/bias/Adam_1/Discriminator/out/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:

"Discriminator/out/bias/Adam_1/readIdentityDiscriminator/out/bias/Adam_1*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *ˇQ9
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
˝
SAdam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam	ApplyAdam=Discriminator/first/Discriminator/firstfully_connected/kernelBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel
Ž
QAdam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdam	ApplyAdam;Discriminator/first/Discriminator/firstfully_connected/bias@Discriminator/first/Discriminator/firstfully_connected/bias/AdamBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:
Ç
UAdam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam	ApplyAdam?Discriminator/second/Discriminator/secondfully_connected/kernelDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
¸
SAdam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdam	ApplyAdam=Discriminator/second/Discriminator/secondfully_connected/biasBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias

.Adam/update_Discriminator/out/kernel/ApplyAdam	ApplyAdamDiscriminator/out/kernelDiscriminator/out/kernel/AdamDiscriminator/out/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
T0*+
_class!
loc:@Discriminator/out/kernel*
use_nesterov( *
_output_shapes
:	*
use_locking( 
ň
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
Ö
Adam/mulMulbeta1_power/read
Adam/beta1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
Ć
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
Ř

Adam/mul_1Mulbeta2_power/read
Adam/beta2R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
Ę
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
â
AdamNoOp^Adam/Assign^Adam/Assign_1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam
V
gradients_1/ShapeShapeLog_2*
out_type0*
_output_shapes
:*
T0
Z
gradients_1/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
{
!gradients_1/Log_2_grad/Reciprocal
Reciprocalsub_1^gradients_1/Fill*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/Log_2_grad/mulMulgradients_1/Fill!gradients_1/Log_2_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
Ŕ
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ť
gradients_1/sub_1_grad/SumSumgradients_1/Log_2_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ż
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
§
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
Ů
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*
_output_shapes
: 
đ
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1
É
8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid1gradients_1/sub_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
ť
=gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp9^gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad9^gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
Ę
Egradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
Ggradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ű
2gradients_1/Discriminator/out_1/MatMul_grad/MatMulMatMulEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(

4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
°
<gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp3^gradients_1/Discriminator/out_1/MatMul_grad/MatMul5^gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1
˝
Dgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/Discriminator/out_1/MatMul_grad/MatMul=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
Fgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	
Ĺ
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Đ
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ň
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
˝
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0

Sgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Đ
\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ó
Mgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
Ogradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
Jgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumMgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
´
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeJgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumOgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ş
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Wgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpO^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeQ^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
Ť
_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeX^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*a
_classW
USloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape
ą
agradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1X^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1

Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ô
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ü
`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
­
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ž
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
¨
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1bgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ć
Tgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapePgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

[gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpS^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeU^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
Š
cgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
Á
egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/AddNAddNagradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N
˝
_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes	
:
á
dgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN`^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad

lgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddNe^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
ngradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrade^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*r
_classh
fdloc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
đ
Ygradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMullgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ů
[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relulgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
Ľ
cgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpZ^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul\^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
Ů
kgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityYgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMuld^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
mgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1Identity[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1d^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

Á
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ě
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
÷
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapekgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ˇ
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0

Qgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
Zgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Kgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualkgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Mgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeroskgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
Hgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumKgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectZgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ž
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeHgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumMgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1\gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
´
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ý
Ugradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpM^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeO^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
Ł
]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeV^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1V^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Đ
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ö
^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapePgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
§
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¨
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
˘
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1`gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ŕ
Rgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ygradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeS^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
Ą
agradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeZ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: 
š
cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1Z^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
gradients_1/AddN_1AddN_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
_output_shapes	
:*
T0*
data_formatNHWC
ß
bgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1^^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad

jgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1c^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
lgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradc^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ę
Wgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMuljgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
°
Ygradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhjgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

agradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulZ^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
Ń
igradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulb^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ď
kgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1b^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

â
(gradients_1/Generator/Tanh_grad/TanhGradTanhGradGenerator/Tanhigradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/Generator/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Ł
9gradients_1/Generator/dense/BiasAdd_grad/tuple/group_depsNoOp)^gradients_1/Generator/Tanh_grad/TanhGrad5^gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad
Ł
Agradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/Generator/Tanh_grad/TanhGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/Generator/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
°
Cgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ń
.gradients_1/Generator/dense/MatMul_grad/MatMulMatMulAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyGenerator/dense/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ů
0gradients_1/Generator/dense/MatMul_grad/MatMul_1MatMul)Generator/forth/Generator/forthleaky_reluAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
¤
8gradients_1/Generator/dense/MatMul_grad/tuple/group_depsNoOp/^gradients_1/Generator/dense/MatMul_grad/MatMul1^gradients_1/Generator/dense/MatMul_grad/MatMul_1
­
@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/Generator/dense/MatMul_grad/MatMul9^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/Generator/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
Bgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/Generator/dense/MatMul_grad/MatMul_19^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/Generator/dense/MatMul_grad/MatMul_1* 
_output_shapes
:

­
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeShape-Generator/forth/Generator/forthleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Á
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
out_type0*
_output_shapes
:*
T0
Â
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Shape@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zerosFillBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
ú
Ggradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqualGreaterEqual-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
Pgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ë
Agradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectSelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
Cgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1SelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/SumSumAgradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectPgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeReshape>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
 
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1SumCgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1Rgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Dgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
Kgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeE^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
ű
Sgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeL^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ugradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1L^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ĺ
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
¸
Tgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulMulSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumSumBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulTgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Mul/Generator/forth/Generator/forthleaky_relu/alphaSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˘
Hgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ë
Ogradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1
ů
Wgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape

Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
gradients_1/AddN_2AddNUgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Ł
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
î
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_2fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ň
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_2hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ë
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ą
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1
Ó
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Ě
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/forth/Generator/forthfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ł
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
î
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
¸
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ů
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ň
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ł
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAddigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ë
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ą
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
Ó
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
ě
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ş
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg
×
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1
¸
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg

Sgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:
˘
Xgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad
Ň
`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape
Ź
bgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ť
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
_output_shapes	
:*
T0
°
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/forth/Generator/forthbatch_normalized/moving_mean/read*
T0*
_output_shapes	
:

agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
ž
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:*
T0
Ä
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:*
T0*i
_class_
][loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
Î
Mgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/forth/Generator/forthfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
ˇ
Ogradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1MatMul)Generator/third/Generator/thirdleaky_relu`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

Wgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1
Š
_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
agradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*b
_classX
VTloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1

gradients_1/AddN_3AddNkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
Ď
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_3:Generator/forth/Generator/forthbatch_normalized/gamma/read*
T0*
_output_shapes	
:
Ö
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_3?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:

_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1
ś
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:
ź
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:
­
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeShape-Generator/third/Generator/thirdleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Á
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
á
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Shape_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zerosFillBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
ú
Ggradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqualGreaterEqual-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
Pgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ę
Agradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectSelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ě
Cgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1SelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/SumSumAgradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectPgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeReshape>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1SumCgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1Rgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Dgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ß
Kgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeE^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1
ű
Sgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeL^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*U
_classK
IGloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape

Ugradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1L^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1

Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Ĺ
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
¸
Tgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulMulSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumSumBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulTgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Mul/Generator/third/Generator/thirdleaky_relu/alphaSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˘
Hgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
Ogradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1
ů
Wgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape*
_output_shapes
: 

Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
gradients_1/AddN_4AddNUgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Ł
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
î
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_4fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ň
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ë
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ą
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1
Ó
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape
Ě
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Ě
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/third/Generator/thirdfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ł
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
î
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
¸
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ň
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAddigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ë
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ą
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1
Ó
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
ě
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:*
T0
Ş
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg
×
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
¸
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:

Sgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:
˘
Xgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad
Ň
`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
bgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ť
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:
°
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/third/Generator/thirdbatch_normalized/moving_mean/read*
_output_shapes	
:*
T0

agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1
ž
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul
Ä
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
Î
Mgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/third/Generator/thirdfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
š
Ogradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1MatMul+Generator/second/Generator/secondleaky_relu`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

Wgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1
Š
_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
agradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:


gradients_1/AddN_5AddNkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
Ď
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_5:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:
Ö
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_5?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt*
T0*
_output_shapes	
:

_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1
ś
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul
ź
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:*
T0
ą
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeShape/Generator/second/Generator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ĺ
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
ă
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Shape_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/zerosFillDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Igradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualGreaterEqual/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Rgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
đ
Cgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectSelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqual_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ň
Egradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1SelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumSumCgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectRgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeReshape@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ś
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1SumEgradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1Tgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Fgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1ReshapeBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ĺ
Mgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_depsNoOpE^gradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeG^gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1

Ugradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyIdentityDgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeN^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Wgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1N^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
É
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
_output_shapes
:*
T0*
out_type0
ž
Vgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulMulUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumSumDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulVgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeReshapeDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0

Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Mul1Generator/second/Generator/secondleaky_relu/alphaUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Xgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
¨
Jgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1ReshapeFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
Qgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_depsNoOpI^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeK^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1

Ygradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeR^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 

[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1R^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*]
_classS
QOloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1
ç
gradients_1/AddN_6AddNWgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ů
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Ľ
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ô
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_6hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ř
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ń
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
§
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1
Ű
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Đ
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeShape8Generator/second/Generator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ľ
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ô
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ž
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumSumVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mulhgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ř
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAddkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ĺ
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ń
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
§
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1
Ű
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ô
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
đ
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/NegNegmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
_output_shapes	
:*
T0
°
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpn^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1U^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg
ß
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitymgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Ŕ
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Negb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:*
T0

Ugradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:
¨
Zgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOpl^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyV^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad
Ú
bgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitykgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
dgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ą
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:
ś
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1Mulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1BGenerator/second/Generator/secondbatch_normalized/moving_mean/read*
T0*
_output_shapes	
:

cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulY^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1
Ć
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Muld^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
Ě
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
Ô
Ogradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulMatMulbgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency<Generator/second/Generator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ť
Qgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1MatMul)Generator/first/Generator/firstleaky_relubgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

Ygradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpP^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulR^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1
ą
agradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulZ^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
cgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1Z^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:


gradients_1/AddN_7AddNmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
Ó
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_7<Generator/second/Generator/secondbatch_normalized/gamma/read*
T0*
_output_shapes	
:
Ú
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_7AGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt*
_output_shapes	
:*
T0

agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpU^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1
ž
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mulb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
_output_shapes	
:*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul
Ä
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:
­
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeShape-Generator/first/Generator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
¸
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
ă
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Shapeagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosFillBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
Ggradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualGreaterEqual-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Pgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ě
Agradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectSelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
î
Cgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1SelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqual@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

>gradients_1/Generator/first/Generator/firstleaky_relu_grad/SumSumAgradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectPgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeReshape>gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
 
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1SumCgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1Rgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Dgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
Kgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeE^gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
ű
Sgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeL^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*U
_classK
IGloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape

Ugradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1L^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
ź
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
¸
Tgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulMulSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumSumBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulTgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Mul/Generator/first/Generator/firstleaky_relu/alphaSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
˘
Hgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
Ogradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1
ů
Wgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape

Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
gradients_1/AddN_8AddNUgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*
N
ł
Sgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:
Ë
Xgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_8T^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad
ç
`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8Y^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
bgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Í
Mgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/first/Generator/firstfully_connected/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d*
transpose_a( *
transpose_b(

Ogradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	d*
transpose_a(

Wgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1
¨
_gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
Ś
agradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d

beta1_power_1/initial_valueConst*'
_class
loc:@Generator/dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 

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
˝
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
w
beta1_power_1/readIdentitybeta1_power_1*
_output_shapes
: *
T0*'
_class
loc:@Generator/dense/bias

beta2_power_1/initial_valueConst*'
_class
loc:@Generator/dense/bias*
valueB
 *wž?*
dtype0*
_output_shapes
: 

beta2_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape: 
˝
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
÷
\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
á
RGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    

LGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zerosFill\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	d
ú
:Generator/first/Generator/firstfully_connected/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	d*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d
ň
AGenerator/first/Generator/firstfully_connected/kernel/Adam/AssignAssign:Generator/first/Generator/firstfully_connected/kernel/AdamLGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
ű
?Generator/first/Generator/firstfully_connected/kernel/Adam/readIdentity:Generator/first/Generator/firstfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d
ů
^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
ă
TGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	d*
T0
ü
<Generator/first/Generator/firstfully_connected/kernel/Adam_1
VariableV2*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d
ř
CGenerator/first/Generator/firstfully_connected/kernel/Adam_1/AssignAssign<Generator/first/Generator/firstfully_connected/kernel/Adam_1NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(
˙
AGenerator/first/Generator/firstfully_connected/kernel/Adam_1/readIdentity<Generator/first/Generator/firstfully_connected/kernel/Adam_1*
_output_shapes
:	d*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
á
JGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB*    
î
8Generator/first/Generator/firstfully_connected/bias/Adam
VariableV2*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ć
?Generator/first/Generator/firstfully_connected/bias/Adam/AssignAssign8Generator/first/Generator/firstfully_connected/bias/AdamJGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:
ń
=Generator/first/Generator/firstfully_connected/bias/Adam/readIdentity8Generator/first/Generator/firstfully_connected/bias/Adam*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
ă
LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
đ
:Generator/first/Generator/firstfully_connected/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
ě
AGenerator/first/Generator/firstfully_connected/bias/Adam_1/AssignAssign:Generator/first/Generator/firstfully_connected/bias/Adam_1LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ő
?Generator/first/Generator/firstfully_connected/bias/Adam_1/readIdentity:Generator/first/Generator/firstfully_connected/bias/Adam_1*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
ű
^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ĺ
TGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

NGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zerosFill^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorTGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:


<Generator/second/Generator/secondfully_connected/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container 
ű
CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignAssign<Generator/second/Generator/secondfully_connected/kernel/AdamNGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel

AGenerator/second/Generator/secondfully_connected/kernel/Adam/readIdentity<Generator/second/Generator/secondfully_connected/kernel/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:

ý
`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"      
ç
VGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    

PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zerosFill`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorVGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0

>Generator/second/Generator/secondfully_connected/kernel/Adam_1
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container 

EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignAssign>Generator/second/Generator/secondfully_connected/kernel/Adam_1PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

CGenerator/second/Generator/secondfully_connected/kernel/Adam_1/readIdentity>Generator/second/Generator/secondfully_connected/kernel/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:

ĺ
LGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB*    
ň
:Generator/second/Generator/secondfully_connected/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
î
AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAssign:Generator/second/Generator/secondfully_connected/bias/AdamLGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
÷
?Generator/second/Generator/secondfully_connected/bias/Adam/readIdentity:Generator/second/Generator/secondfully_connected/bias/Adam*
_output_shapes	
:*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
ç
NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB*    
ô
<Generator/second/Generator/secondfully_connected/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container 
ô
CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignAssign<Generator/second/Generator/secondfully_connected/bias/Adam_1NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(
ű
AGenerator/second/Generator/secondfully_connected/bias/Adam_1/readIdentity<Generator/second/Generator/secondfully_connected/bias/Adam_1*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:
é
NGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ö
<Generator/second/Generator/secondbatch_normalized/gamma/Adam
VariableV2*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
ö
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam/AssignAssign<Generator/second/Generator/secondbatch_normalized/gamma/AdamNGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:
ý
AGenerator/second/Generator/secondbatch_normalized/gamma/Adam/readIdentity<Generator/second/Generator/secondbatch_normalized/gamma/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:
ë
PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ř
>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1
VariableV2*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
ü
EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignAssign>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:

CGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/readIdentity>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:
ç
MGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB*    *
dtype0*
_output_shapes	
:
ô
;Generator/second/Generator/secondbatch_normalized/beta/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
ň
BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignAssign;Generator/second/Generator/secondbatch_normalized/beta/AdamMGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ú
@Generator/second/Generator/secondbatch_normalized/beta/Adam/readIdentity;Generator/second/Generator/secondbatch_normalized/beta/Adam*
_output_shapes	
:*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
é
OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zerosConst*
_output_shapes	
:*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB*    *
dtype0
ö
=Generator/second/Generator/secondbatch_normalized/beta/Adam_1
VariableV2*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ř
DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignAssign=Generator/second/Generator/secondbatch_normalized/beta/Adam_1OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ţ
BGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/readIdentity=Generator/second/Generator/secondbatch_normalized/beta/Adam_1*
_output_shapes	
:*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
÷
\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
á
RGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

LGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zerosFill\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0
ü
:Generator/third/Generator/thirdfully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:

ó
AGenerator/third/Generator/thirdfully_connected/kernel/Adam/AssignAssign:Generator/third/Generator/thirdfully_connected/kernel/AdamLGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ü
?Generator/third/Generator/thirdfully_connected/kernel/Adam/readIdentity:Generator/third/Generator/thirdfully_connected/kernel/Adam* 
_output_shapes
:
*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
ů
^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ă
TGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0
ţ
<Generator/third/Generator/thirdfully_connected/kernel/Adam_1
VariableV2* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
*
dtype0
ů
CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignAssign<Generator/third/Generator/thirdfully_connected/kernel/Adam_1NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

AGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/readIdentity<Generator/third/Generator/thirdfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:

á
JGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
î
8Generator/third/Generator/thirdfully_connected/bias/Adam
VariableV2*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ć
?Generator/third/Generator/thirdfully_connected/bias/Adam/AssignAssign8Generator/third/Generator/thirdfully_connected/bias/AdamJGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ń
=Generator/third/Generator/thirdfully_connected/bias/Adam/readIdentity8Generator/third/Generator/thirdfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:
ă
LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
đ
:Generator/third/Generator/thirdfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:
ě
AGenerator/third/Generator/thirdfully_connected/bias/Adam_1/AssignAssign:Generator/third/Generator/thirdfully_connected/bias/Adam_1LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:
ő
?Generator/third/Generator/thirdfully_connected/bias/Adam_1/readIdentity:Generator/third/Generator/thirdfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:
ĺ
LGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zerosConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ň
:Generator/third/Generator/thirdbatch_normalized/gamma/Adam
VariableV2*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
î
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAssign:Generator/third/Generator/thirdbatch_normalized/gamma/AdamLGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:
÷
?Generator/third/Generator/thirdbatch_normalized/gamma/Adam/readIdentity:Generator/third/Generator/thirdbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:
ç
NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB*    
ô
<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:
ô
CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignAssign<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:
ű
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/readIdentity<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:
ă
KGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB*    *
dtype0*
_output_shapes	
:
đ
9Generator/third/Generator/thirdbatch_normalized/beta/Adam
VariableV2*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ę
@Generator/third/Generator/thirdbatch_normalized/beta/Adam/AssignAssign9Generator/third/Generator/thirdbatch_normalized/beta/AdamKGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ô
>Generator/third/Generator/thirdbatch_normalized/beta/Adam/readIdentity9Generator/third/Generator/thirdbatch_normalized/beta/Adam*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:*
T0
ĺ
MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB*    *
dtype0*
_output_shapes	
:
ň
;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:*
dtype0
đ
BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignAssign;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ř
@Generator/third/Generator/thirdbatch_normalized/beta/Adam_1/readIdentity;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1*
_output_shapes	
:*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
÷
\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
á
RGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

LGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0* 
_output_shapes
:

ü
:Generator/forth/Generator/forthfully_connected/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container 
ó
AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAssign:Generator/forth/Generator/forthfully_connected/kernel/AdamLGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:

ü
?Generator/forth/Generator/forthfully_connected/kernel/Adam/readIdentity:Generator/forth/Generator/forthfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:

ů
^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ă
TGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0
ţ
<Generator/forth/Generator/forthfully_connected/kernel/Adam_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
ů
CGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/AssignAssign<Generator/forth/Generator/forthfully_connected/kernel/Adam_1NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:


AGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/readIdentity<Generator/forth/Generator/forthfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:

í
ZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:*
dtype0
Ý
PGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 

JGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zerosFillZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorPGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/Const*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0
î
8Generator/forth/Generator/forthfully_connected/bias/Adam
VariableV2*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ć
?Generator/forth/Generator/forthfully_connected/bias/Adam/AssignAssign8Generator/forth/Generator/forthfully_connected/bias/AdamJGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:
ń
=Generator/forth/Generator/forthfully_connected/bias/Adam/readIdentity8Generator/forth/Generator/forthfully_connected/bias/Adam*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
ď
\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
ß
RGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 

LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0
đ
:Generator/forth/Generator/forthfully_connected/bias/Adam_1
VariableV2*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ě
AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAssign:Generator/forth/Generator/forthfully_connected/bias/Adam_1LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:
ő
?Generator/forth/Generator/forthfully_connected/bias/Adam_1/readIdentity:Generator/forth/Generator/forthfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:
ń
\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:*
dtype0*
_output_shapes
:
á
RGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 

LGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zerosFill\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/Const*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:*
T0
ň
:Generator/forth/Generator/forthbatch_normalized/gamma/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container 
î
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAssign:Generator/forth/Generator/forthbatch_normalized/gamma/AdamLGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:
÷
?Generator/forth/Generator/forthbatch_normalized/gamma/Adam/readIdentity:Generator/forth/Generator/forthbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:
ó
^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:
ă
TGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 

NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:
ô
<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
ô
CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignAssign<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(
ű
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/readIdentity<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:
ď
[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:*
dtype0*
_output_shapes
:
ß
QGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 

KGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zerosFill[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorQGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/Const*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:*
T0
đ
9Generator/forth/Generator/forthbatch_normalized/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:
ę
@Generator/forth/Generator/forthbatch_normalized/beta/Adam/AssignAssign9Generator/forth/Generator/forthbatch_normalized/beta/AdamKGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ô
>Generator/forth/Generator/forthbatch_normalized/beta/Adam/readIdentity9Generator/forth/Generator/forthbatch_normalized/beta/Adam*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:
ń
]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:*
dtype0*
_output_shapes
:
á
SGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    

MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:
ň
;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:
đ
BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignAssign;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ř
@Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/readIdentity;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:
š
=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ł
3Generator/dense/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-Generator/dense/kernel/Adam/Initializer/zerosFill=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3Generator/dense/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@Generator/dense/kernel*

index_type0* 
_output_shapes
:

ž
Generator/dense/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:

÷
"Generator/dense/kernel/Adam/AssignAssignGenerator/dense/kernel/Adam-Generator/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:


 Generator/dense/kernel/Adam/readIdentityGenerator/dense/kernel/Adam* 
_output_shapes
:
*
T0*)
_class
loc:@Generator/dense/kernel
ť
?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0
Ľ
5Generator/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0

/Generator/dense/kernel/Adam_1/Initializer/zerosFill?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5Generator/dense/kernel/Adam_1/Initializer/zeros/Const*)
_class
loc:@Generator/dense/kernel*

index_type0* 
_output_shapes
:
*
T0
Ŕ
Generator/dense/kernel/Adam_1
VariableV2*)
_class
loc:@Generator/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ý
$Generator/dense/kernel/Adam_1/AssignAssignGenerator/dense/kernel/Adam_1/Generator/dense/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel
Ł
"Generator/dense/kernel/Adam_1/readIdentityGenerator/dense/kernel/Adam_1* 
_output_shapes
:
*
T0*)
_class
loc:@Generator/dense/kernel
Ł
+Generator/dense/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*'
_class
loc:@Generator/dense/bias*
valueB*    *
dtype0
°
Generator/dense/bias/Adam
VariableV2*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ę
 Generator/dense/bias/Adam/AssignAssignGenerator/dense/bias/Adam+Generator/dense/bias/Adam/Initializer/zeros*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

Generator/dense/bias/Adam/readIdentityGenerator/dense/bias/Adam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:
Ľ
-Generator/dense/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
˛
Generator/dense/bias/Adam_1
VariableV2*'
_class
loc:@Generator/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
đ
"Generator/dense/bias/Adam_1/AssignAssignGenerator/dense/bias/Adam_1-Generator/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:

 Generator/dense/bias/Adam_1/readIdentityGenerator/dense/bias/Adam_1*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:*
T0
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
ó
MAdam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/first/Generator/firstfully_connected/kernel:Generator/first/Generator/firstfully_connected/kernel/Adam<Generator/first/Generator/firstfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
use_nesterov( *
_output_shapes
:	d
ć
KAdam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdam	ApplyAdam3Generator/first/Generator/firstfully_connected/bias8Generator/first/Generator/firstfully_connected/bias/Adam:Generator/first/Generator/firstfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0

OAdam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdam	ApplyAdam7Generator/second/Generator/secondfully_connected/kernel<Generator/second/Generator/secondfully_connected/kernel/Adam>Generator/second/Generator/secondfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( 
ň
MAdam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdam	ApplyAdam5Generator/second/Generator/secondfully_connected/bias:Generator/second/Generator/secondfully_connected/bias/Adam<Generator/second/Generator/secondfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0

OAdam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdam	ApplyAdam7Generator/second/Generator/secondbatch_normalized/gamma<Generator/second/Generator/secondbatch_normalized/gamma/Adam>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:
ü
NAdam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdam	ApplyAdam6Generator/second/Generator/secondbatch_normalized/beta;Generator/second/Generator/secondbatch_normalized/beta/Adam=Generator/second/Generator/secondbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:
ô
MAdam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdfully_connected/kernel:Generator/third/Generator/thirdfully_connected/kernel/Adam<Generator/third/Generator/thirdfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
use_nesterov( * 
_output_shapes
:

ć
KAdam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdam	ApplyAdam3Generator/third/Generator/thirdfully_connected/bias8Generator/third/Generator/thirdfully_connected/bias/Adam:Generator/third/Generator/thirdfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
use_nesterov( 
÷
MAdam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdbatch_normalized/gamma:Generator/third/Generator/thirdbatch_normalized/gamma/Adam<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
đ
LAdam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/third/Generator/thirdbatch_normalized/beta9Generator/third/Generator/thirdbatch_normalized/beta/Adam;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:*
use_locking( 
ô
MAdam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthfully_connected/kernel:Generator/forth/Generator/forthfully_connected/kernel/Adam<Generator/forth/Generator/forthfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0
ć
KAdam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdam	ApplyAdam3Generator/forth/Generator/forthfully_connected/bias8Generator/forth/Generator/forthfully_connected/bias/Adam:Generator/forth/Generator/forthfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias
÷
MAdam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthbatch_normalized/gamma:Generator/forth/Generator/forthbatch_normalized/gamma/Adam<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:
đ
LAdam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/forth/Generator/forthbatch_normalized/beta9Generator/forth/Generator/forthbatch_normalized/beta/Adam;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_locking( *
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:
ş
.Adam_1/update_Generator/dense/kernel/ApplyAdam	ApplyAdamGenerator/dense/kernelGenerator/dense/kernel/AdamGenerator/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@Generator/dense/kernel*
use_nesterov( * 
_output_shapes
:

Ź
,Adam_1/update_Generator/dense/bias/ApplyAdam	ApplyAdamGenerator/dense/biasGenerator/dense/bias/AdamGenerator/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*'
_class
loc:@Generator/dense/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
ş


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*'
_class
loc:@Generator/dense/bias
Ľ
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
ź

Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
Š
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
í	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: "ěşM˙P     ĆQ×	|	#ŽýÖAJ 
˘
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

2	
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
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.12.02v1.12.0-0-ga6d8ffae09ďČ
r
Generator/noisePlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d*
shape:˙˙˙˙˙˙˙˙˙d
ń
VGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d      
ă
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&ž*
dtype0
ă
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *_&>*
dtype0*
_output_shapes
: 
á
^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/shape*

seed*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
seed2*
dtype0*
_output_shapes
:	d
ň
TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/maxTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel

TGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/sub*
_output_shapes
:	d*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
÷
PGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniformAddTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/mulTGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform/min*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d*
T0
ő
5Generator/first/Generator/firstfully_connected/kernel
VariableV2*
_output_shapes
:	d*
shared_name *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d*
dtype0
ě
<Generator/first/Generator/firstfully_connected/kernel/AssignAssign5Generator/first/Generator/firstfully_connected/kernelPGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0
ń
:Generator/first/Generator/firstfully_connected/kernel/readIdentity5Generator/first/Generator/firstfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
_output_shapes
:	d
Ü
EGenerator/first/Generator/firstfully_connected/bias/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
é
3Generator/first/Generator/firstfully_connected/bias
VariableV2*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
×
:Generator/first/Generator/firstfully_connected/bias/AssignAssign3Generator/first/Generator/firstfully_connected/biasEGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(
ç
8Generator/first/Generator/firstfully_connected/bias/readIdentity3Generator/first/Generator/firstfully_connected/bias*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
_output_shapes	
:*
T0
ĺ
5Generator/first/Generator/firstfully_connected/MatMulMatMulGenerator/noise:Generator/first/Generator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ü
6Generator/first/Generator/firstfully_connected/BiasAddBiasAdd5Generator/first/Generator/firstfully_connected/MatMul8Generator/first/Generator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
/Generator/first/Generator/firstleaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Đ
-Generator/first/Generator/firstleaky_relu/mulMul/Generator/first/Generator/firstleaky_relu/alpha6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
)Generator/first/Generator/firstleaky_reluMaximum-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
XGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"      
ç
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   ž*
dtype0
ç
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *   >*
dtype0*
_output_shapes
: 
č
`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformXGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
seed2
ú
VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/subSubVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/maxVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel

VGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulMul`Generator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/sub*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
*
T0

RGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniformAddVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/mulVGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:

ű
7Generator/second/Generator/secondfully_connected/kernel
VariableV2*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ő
>Generator/second/Generator/secondfully_connected/kernel/AssignAssign7Generator/second/Generator/secondfully_connected/kernelRGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:

ř
<Generator/second/Generator/secondfully_connected/kernel/readIdentity7Generator/second/Generator/secondfully_connected/kernel*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:
*
T0
ŕ
GGenerator/second/Generator/secondfully_connected/bias/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
í
5Generator/second/Generator/secondfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:
ß
<Generator/second/Generator/secondfully_connected/bias/AssignAssign5Generator/second/Generator/secondfully_connected/biasGGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:
í
:Generator/second/Generator/secondfully_connected/bias/readIdentity5Generator/second/Generator/secondfully_connected/bias*
_output_shapes	
:*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias

7Generator/second/Generator/secondfully_connected/MatMulMatMul)Generator/first/Generator/firstleaky_relu<Generator/second/Generator/secondfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

8Generator/second/Generator/secondfully_connected/BiasAddBiasAdd7Generator/second/Generator/secondfully_connected/MatMul:Generator/second/Generator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ă
HGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/onesConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB*  ?*
dtype0*
_output_shapes	
:
ń
7Generator/second/Generator/secondbatch_normalized/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container 
ć
>Generator/second/Generator/secondbatch_normalized/gamma/AssignAssign7Generator/second/Generator/secondbatch_normalized/gammaHGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:
ó
<Generator/second/Generator/secondbatch_normalized/gamma/readIdentity7Generator/second/Generator/secondbatch_normalized/gamma*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
_output_shapes	
:
â
HGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB*    *
dtype0*
_output_shapes	
:
ď
6Generator/second/Generator/secondbatch_normalized/beta
VariableV2*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ă
=Generator/second/Generator/secondbatch_normalized/beta/AssignAssign6Generator/second/Generator/secondbatch_normalized/betaHGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
đ
;Generator/second/Generator/secondbatch_normalized/beta/readIdentity6Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
đ
OGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zerosConst*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ý
=Generator/second/Generator/secondbatch_normalized/moving_mean
VariableV2*
shared_name *P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
	container *
shape:*
dtype0*
_output_shapes	
:
˙
DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignAssign=Generator/second/Generator/secondbatch_normalized/moving_meanOGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(

BGenerator/second/Generator/secondbatch_normalized/moving_mean/readIdentity=Generator/second/Generator/secondbatch_normalized/moving_mean*
T0*P
_classF
DBloc:@Generator/second/Generator/secondbatch_normalized/moving_mean*
_output_shapes	
:
÷
RGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/onesConst*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:

AGenerator/second/Generator/secondbatch_normalized/moving_variance
VariableV2*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

HGenerator/second/Generator/secondbatch_normalized/moving_variance/AssignAssignAGenerator/second/Generator/secondbatch_normalized/moving_varianceRGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/ones*
T0*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:*
use_locking(

FGenerator/second/Generator/secondbatch_normalized/moving_variance/readIdentityAGenerator/second/Generator/secondbatch_normalized/moving_variance*T
_classJ
HFloc:@Generator/second/Generator/secondbatch_normalized/moving_variance*
_output_shapes	
:*
T0

AGenerator/second/Generator/secondbatch_normalized/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o:*
dtype0
÷
?Generator/second/Generator/secondbatch_normalized/batchnorm/addAddFGenerator/second/Generator/secondbatch_normalized/moving_variance/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/add/y*
_output_shapes	
:*
T0
ą
AGenerator/second/Generator/secondbatch_normalized/batchnorm/RsqrtRsqrt?Generator/second/Generator/secondbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:
í
?Generator/second/Generator/secondbatch_normalized/batchnorm/mulMulAGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt<Generator/second/Generator/secondbatch_normalized/gamma/read*
_output_shapes	
:*
T0
ö
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAdd?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ó
AGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2MulBGenerator/second/Generator/secondbatch_normalized/moving_mean/read?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:
ě
?Generator/second/Generator/secondbatch_normalized/batchnorm/subSub;Generator/second/Generator/secondbatch_normalized/beta/readAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:
˙
AGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1AddAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1?Generator/second/Generator/secondbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
v
1Generator/second/Generator/secondleaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
ß
/Generator/second/Generator/secondleaky_relu/mulMul1Generator/second/Generator/secondleaky_relu/alphaAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
+Generator/second/Generator/secondleaky_reluMaximum/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
VGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0
ă
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *óľ˝*
dtype0*
_output_shapes
: 
ă
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *óľ=*
dtype0*
_output_shapes
: 
â
^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
seed2B
ň
TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/maxTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
_output_shapes
: 

TGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:

ř
PGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniformAddTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/mulTGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:

÷
5Generator/third/Generator/thirdfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:

í
<Generator/third/Generator/thirdfully_connected/kernel/AssignAssign5Generator/third/Generator/thirdfully_connected/kernelPGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:

ň
:Generator/third/Generator/thirdfully_connected/kernel/readIdentity5Generator/third/Generator/thirdfully_connected/kernel*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:

Ü
EGenerator/third/Generator/thirdfully_connected/bias/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
é
3Generator/third/Generator/thirdfully_connected/bias
VariableV2*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:*
dtype0
×
:Generator/third/Generator/thirdfully_connected/bias/AssignAssign3Generator/third/Generator/thirdfully_connected/biasEGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
ç
8Generator/third/Generator/thirdfully_connected/bias/readIdentity3Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias

5Generator/third/Generator/thirdfully_connected/MatMulMatMul+Generator/second/Generator/secondleaky_relu:Generator/third/Generator/thirdfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
ü
6Generator/third/Generator/thirdfully_connected/BiasAddBiasAdd5Generator/third/Generator/thirdfully_connected/MatMul8Generator/third/Generator/thirdfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
FGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB*  ?
í
5Generator/third/Generator/thirdbatch_normalized/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:
Ţ
<Generator/third/Generator/thirdbatch_normalized/gamma/AssignAssign5Generator/third/Generator/thirdbatch_normalized/gammaFGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
í
:Generator/third/Generator/thirdbatch_normalized/gamma/readIdentity5Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
Ţ
FGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zerosConst*
_output_shapes	
:*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB*    *
dtype0
ë
4Generator/third/Generator/thirdbatch_normalized/beta
VariableV2*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ű
;Generator/third/Generator/thirdbatch_normalized/beta/AssignAssign4Generator/third/Generator/thirdbatch_normalized/betaFGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ę
9Generator/third/Generator/thirdbatch_normalized/beta/readIdentity4Generator/third/Generator/thirdbatch_normalized/beta*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:
ě
MGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zerosConst*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
valueB*    *
dtype0*
_output_shapes	
:
ů
;Generator/third/Generator/thirdbatch_normalized/moving_mean
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean
÷
BGenerator/third/Generator/thirdbatch_normalized/moving_mean/AssignAssign;Generator/third/Generator/thirdbatch_normalized/moving_meanMGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zeros*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:*
use_locking(
˙
@Generator/third/Generator/thirdbatch_normalized/moving_mean/readIdentity;Generator/third/Generator/thirdbatch_normalized/moving_mean*
T0*N
_classD
B@loc:@Generator/third/Generator/thirdbatch_normalized/moving_mean*
_output_shapes	
:
ó
PGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/onesConst*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
valueB*  ?*
dtype0*
_output_shapes	
:

?Generator/third/Generator/thirdbatch_normalized/moving_variance
VariableV2*
shared_name *R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:

FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignAssign?Generator/third/Generator/thirdbatch_normalized/moving_variancePGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones*
use_locking(*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
validate_shape(*
_output_shapes	
:

DGenerator/third/Generator/thirdbatch_normalized/moving_variance/readIdentity?Generator/third/Generator/thirdbatch_normalized/moving_variance*
T0*R
_classH
FDloc:@Generator/third/Generator/thirdbatch_normalized/moving_variance*
_output_shapes	
:

?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/yConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
ń
=Generator/third/Generator/thirdbatch_normalized/batchnorm/addAddDGenerator/third/Generator/thirdbatch_normalized/moving_variance/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:
­
?Generator/third/Generator/thirdbatch_normalized/batchnorm/RsqrtRsqrt=Generator/third/Generator/thirdbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:
ç
=Generator/third/Generator/thirdbatch_normalized/batchnorm/mulMul?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:
đ
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAdd=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2Mul@Generator/third/Generator/thirdbatch_normalized/moving_mean/read=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
_output_shapes	
:*
T0
ć
=Generator/third/Generator/thirdbatch_normalized/batchnorm/subSub9Generator/third/Generator/thirdbatch_normalized/beta/read?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:
ů
?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1Add?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
t
/Generator/third/Generator/thirdleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *ÍĚL>*
dtype0
Ů
-Generator/third/Generator/thirdleaky_relu/mulMul/Generator/third/Generator/thirdleaky_relu/alpha?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
)Generator/third/Generator/thirdleaky_reluMaximum-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
VGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shapeConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ă
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/minConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  ˝*
dtype0*
_output_shapes
: 
ă
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *  =*
dtype0*
_output_shapes
: 
â
^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniformVGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/shape*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
seed2m*
dtype0* 
_output_shapes
:
*

seed
ň
TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/subSubTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/maxTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
_output_shapes
: 

TGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulMul^Generator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/RandomUniformTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/sub*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:

ř
PGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniformAddTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/mulTGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
÷
5Generator/forth/Generator/forthfully_connected/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
í
<Generator/forth/Generator/forthfully_connected/kernel/AssignAssign5Generator/forth/Generator/forthfully_connected/kernelPGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:

ň
:Generator/forth/Generator/forthfully_connected/kernel/readIdentity5Generator/forth/Generator/forthfully_connected/kernel*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:
*
T0
č
UGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
Ř
KGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
ń
EGenerator/forth/Generator/forthfully_connected/bias/Initializer/zerosFillUGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/shape_as_tensorKGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros/Const*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:
é
3Generator/forth/Generator/forthfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:
×
:Generator/forth/Generator/forthfully_connected/bias/AssignAssign3Generator/forth/Generator/forthfully_connected/biasEGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(
ç
8Generator/forth/Generator/forthfully_connected/bias/readIdentity3Generator/forth/Generator/forthfully_connected/bias*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:
˙
5Generator/forth/Generator/forthfully_connected/MatMulMatMul)Generator/third/Generator/thirdleaky_relu:Generator/forth/Generator/forthfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
ü
6Generator/forth/Generator/forthfully_connected/BiasAddBiasAdd5Generator/forth/Generator/forthfully_connected/MatMul8Generator/forth/Generator/forthfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
VGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:*
dtype0*
_output_shapes
:
Ű
LGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ö
FGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/onesFillVGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones/Const*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:*
T0
í
5Generator/forth/Generator/forthbatch_normalized/gamma
VariableV2*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ţ
<Generator/forth/Generator/forthbatch_normalized/gamma/AssignAssign5Generator/forth/Generator/forthbatch_normalized/gammaFGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
í
:Generator/forth/Generator/forthbatch_normalized/gamma/readIdentity5Generator/forth/Generator/forthbatch_normalized/gamma*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:
ę
VGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:*
dtype0*
_output_shapes
:
Ú
LGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
ő
FGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zerosFillVGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/shape_as_tensorLGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:
ë
4Generator/forth/Generator/forthbatch_normalized/beta
VariableV2*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:*
dtype0
Ű
;Generator/forth/Generator/forthbatch_normalized/beta/AssignAssign4Generator/forth/Generator/forthbatch_normalized/betaFGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ę
9Generator/forth/Generator/forthbatch_normalized/beta/readIdentity4Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
ř
]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorConst*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB:*
dtype0*
_output_shapes
:
č
SGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
valueB
 *    

MGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros/Const*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*

index_type0*
_output_shapes	
:
ů
;Generator/forth/Generator/forthbatch_normalized/moving_mean
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
	container 
÷
BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignAssign;Generator/forth/Generator/forthbatch_normalized/moving_meanMGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
validate_shape(*
_output_shapes	
:
˙
@Generator/forth/Generator/forthbatch_normalized/moving_mean/readIdentity;Generator/forth/Generator/forthbatch_normalized/moving_mean*
T0*N
_classD
B@loc:@Generator/forth/Generator/forthbatch_normalized/moving_mean*
_output_shapes	
:
˙
`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorConst*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB:*
dtype0*
_output_shapes
:
ď
VGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/ConstConst*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
valueB
 *  ?*
dtype0*
_output_shapes
: 

PGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/onesFill`Generator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/shape_as_tensorVGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones/Const*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*

index_type0*
_output_shapes	
:

?Generator/forth/Generator/forthbatch_normalized/moving_variance
VariableV2*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 

FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignAssign?Generator/forth/Generator/forthbatch_normalized/moving_variancePGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
validate_shape(

DGenerator/forth/Generator/forthbatch_normalized/moving_variance/readIdentity?Generator/forth/Generator/forthbatch_normalized/moving_variance*R
_classH
FDloc:@Generator/forth/Generator/forthbatch_normalized/moving_variance*
_output_shapes	
:*
T0

?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o:*
dtype0
ń
=Generator/forth/Generator/forthbatch_normalized/batchnorm/addAddDGenerator/forth/Generator/forthbatch_normalized/moving_variance/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/add/y*
T0*
_output_shapes	
:
­
?Generator/forth/Generator/forthbatch_normalized/batchnorm/RsqrtRsqrt=Generator/forth/Generator/forthbatch_normalized/batchnorm/add*
T0*
_output_shapes	
:
ç
=Generator/forth/Generator/forthbatch_normalized/batchnorm/mulMul?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt:Generator/forth/Generator/forthbatch_normalized/gamma/read*
_output_shapes	
:*
T0
đ
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAdd=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2Mul@Generator/forth/Generator/forthbatch_normalized/moving_mean/read=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
_output_shapes	
:*
T0
ć
=Generator/forth/Generator/forthbatch_normalized/batchnorm/subSub9Generator/forth/Generator/forthbatch_normalized/beta/read?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2*
T0*
_output_shapes	
:
ů
?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1Add?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/sub*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
/Generator/forth/Generator/forthleaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
Ů
-Generator/forth/Generator/forthleaky_relu/mulMul/Generator/forth/Generator/forthleaky_relu/alpha?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
)Generator/forth/Generator/forthleaky_reluMaximum-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
7Generator/dense/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ľ
5Generator/dense/kernel/Initializer/random_uniform/minConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *zők˝*
dtype0*
_output_shapes
: 
Ľ
5Generator/dense/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *zők=*
dtype0*
_output_shapes
: 

?Generator/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7Generator/dense/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed*
T0*)
_class
loc:@Generator/dense/kernel*
seed2˘
ö
5Generator/dense/kernel/Initializer/random_uniform/subSub5Generator/dense/kernel/Initializer/random_uniform/max5Generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@Generator/dense/kernel*
_output_shapes
: 

5Generator/dense/kernel/Initializer/random_uniform/mulMul?Generator/dense/kernel/Initializer/random_uniform/RandomUniform5Generator/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:

ü
1Generator/dense/kernel/Initializer/random_uniformAdd5Generator/dense/kernel/Initializer/random_uniform/mul5Generator/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:

š
Generator/dense/kernel
VariableV2*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ń
Generator/dense/kernel/AssignAssignGenerator/dense/kernel1Generator/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:


Generator/dense/kernel/readIdentityGenerator/dense/kernel*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
*
T0

&Generator/dense/bias/Initializer/zerosConst*
_output_shapes	
:*'
_class
loc:@Generator/dense/bias*
valueB*    *
dtype0
Ť
Generator/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:
Ű
Generator/dense/bias/AssignAssignGenerator/dense/bias&Generator/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:

Generator/dense/bias/readIdentityGenerator/dense/bias*
_output_shapes	
:*
T0*'
_class
loc:@Generator/dense/bias
Á
Generator/dense/MatMulMatMul)Generator/forth/Generator/forthleaky_reluGenerator/dense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

Generator/dense/BiasAddBiasAddGenerator/dense/MatMulGenerator/dense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
b
Generator/TanhTanhGenerator/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
w
Discriminator/realPlaceholder*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙

^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shapeConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ó
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/minConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HY˝*
dtype0*
_output_shapes
: 
ó
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/maxConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
ű
fDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform^Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/shape*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
seed2´*
dtype0* 
_output_shapes
:
*

seed*
T0

\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/subSub\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/max\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
_output_shapes
: 
Ś
\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mulMulfDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/RandomUniform\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel

XDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniformAdd\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/mul\Discriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform/min*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:


=Discriminator/first/Discriminator/firstfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:


DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignAssign=Discriminator/first/Discriminator/firstfully_connected/kernelXDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:


BDiscriminator/first/Discriminator/firstfully_connected/kernel/readIdentity=Discriminator/first/Discriminator/firstfully_connected/kernel*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:

ě
MDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ů
;Discriminator/first/Discriminator/firstfully_connected/bias
VariableV2*
_output_shapes	
:*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:*
dtype0
÷
BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignAssign;Discriminator/first/Discriminator/firstfully_connected/biasMDiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
˙
@Discriminator/first/Discriminator/firstfully_connected/bias/readIdentity;Discriminator/first/Discriminator/firstfully_connected/bias*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:
ř
=Discriminator/first/Discriminator/firstfully_connected/MatMulMatMulDiscriminator/realBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

>Discriminator/first/Discriminator/firstfully_connected/BiasAddBiasAdd=Discriminator/first/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
|
7Discriminator/first/Discriminator/firstleaky_relu/alphaConst*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL>
č
5Discriminator/first/Discriminator/firstleaky_relu/mulMul7Discriminator/first/Discriminator/firstleaky_relu/alpha>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ć
1Discriminator/first/Discriminator/firstleaky_reluMaximum5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      
÷
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *óľ˝*
dtype0
÷
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *óľ=*
dtype0

hDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniformRandomUniform`Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/shape*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
seed2Ç*
dtype0* 
_output_shapes
:
*

seed

^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/subSub^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/max^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
_output_shapes
: 
Ž
^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mulMulhDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/RandomUniform^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/sub*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
*
T0
 
ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniformAdd^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/mul^Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel

?Discriminator/second/Discriminator/secondfully_connected/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:


FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignAssign?Discriminator/second/Discriminator/secondfully_connected/kernelZDiscriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(

DDiscriminator/second/Discriminator/secondfully_connected/kernel/readIdentity?Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:
*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
đ
ODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ý
=Discriminator/second/Discriminator/secondfully_connected/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:
˙
DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignAssign=Discriminator/second/Discriminator/secondfully_connected/biasODiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:

BDiscriminator/second/Discriminator/secondfully_connected/bias/readIdentity=Discriminator/second/Discriminator/secondfully_connected/bias*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:*
T0

?Discriminator/second/Discriminator/secondfully_connected/MatMulMatMul1Discriminator/first/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

@Discriminator/second/Discriminator/secondfully_connected/BiasAddBiasAdd?Discriminator/second/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
~
9Discriminator/second/Discriminator/secondleaky_relu/alphaConst*
_output_shapes
: *
valueB
 *ÍĚL>*
dtype0
î
7Discriminator/second/Discriminator/secondleaky_relu/mulMul9Discriminator/second/Discriminator/secondleaky_relu/alpha@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
3Discriminator/second/Discriminator/secondleaky_reluMaximum7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
9Discriminator/out/kernel/Initializer/random_uniform/shapeConst*+
_class!
loc:@Discriminator/out/kernel*
valueB"      *
dtype0*
_output_shapes
:
Š
7Discriminator/out/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Ivž
Š
7Discriminator/out/kernel/Initializer/random_uniform/maxConst*+
_class!
loc:@Discriminator/out/kernel*
valueB
 *Iv>*
dtype0*
_output_shapes
: 

ADiscriminator/out/kernel/Initializer/random_uniform/RandomUniformRandomUniform9Discriminator/out/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed*
T0*+
_class!
loc:@Discriminator/out/kernel*
seed2Ú
ţ
7Discriminator/out/kernel/Initializer/random_uniform/subSub7Discriminator/out/kernel/Initializer/random_uniform/max7Discriminator/out/kernel/Initializer/random_uniform/min*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
: *
T0

7Discriminator/out/kernel/Initializer/random_uniform/mulMulADiscriminator/out/kernel/Initializer/random_uniform/RandomUniform7Discriminator/out/kernel/Initializer/random_uniform/sub*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	*
T0

3Discriminator/out/kernel/Initializer/random_uniformAdd7Discriminator/out/kernel/Initializer/random_uniform/mul7Discriminator/out/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	
ť
Discriminator/out/kernel
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	
ř
Discriminator/out/kernel/AssignAssignDiscriminator/out/kernel3Discriminator/out/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	

Discriminator/out/kernel/readIdentityDiscriminator/out/kernel*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	*
T0
 
(Discriminator/out/bias/Initializer/zerosConst*
_output_shapes
:*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0
­
Discriminator/out/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias
â
Discriminator/out/bias/AssignAssignDiscriminator/out/bias(Discriminator/out/bias/Initializer/zeros*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

Discriminator/out/bias/readIdentityDiscriminator/out/bias*
T0*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:
Î
Discriminator/out/MatMulMatMul3Discriminator/second/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
¤
Discriminator/out/BiasAddBiasAddDiscriminator/out/MatMulDiscriminator/out/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
q
Discriminator/out/SigmoidSigmoidDiscriminator/out/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ö
?Discriminator/first_1/Discriminator/firstfully_connected/MatMulMatMulGenerator/TanhBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

@Discriminator/first_1/Discriminator/firstfully_connected/BiasAddBiasAdd?Discriminator/first_1/Discriminator/firstfully_connected/MatMul@Discriminator/first/Discriminator/firstfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
9Discriminator/first_1/Discriminator/firstleaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
î
7Discriminator/first_1/Discriminator/firstleaky_relu/mulMul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
3Discriminator/first_1/Discriminator/firstleaky_reluMaximum7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

ADiscriminator/second_1/Discriminator/secondfully_connected/MatMulMatMul3Discriminator/first_1/Discriminator/firstleaky_reluDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

BDiscriminator/second_1/Discriminator/secondfully_connected/BiasAddBiasAddADiscriminator/second_1/Discriminator/secondfully_connected/MatMulBDiscriminator/second/Discriminator/secondfully_connected/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

;Discriminator/second_1/Discriminator/secondleaky_relu/alphaConst*
valueB
 *ÍĚL>*
dtype0*
_output_shapes
: 
ô
9Discriminator/second_1/Discriminator/secondleaky_relu/mulMul;Discriminator/second_1/Discriminator/secondleaky_relu/alphaBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ň
5Discriminator/second_1/Discriminator/secondleaky_reluMaximum9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ň
Discriminator/out_1/MatMulMatMul5Discriminator/second_1/Discriminator/secondleaky_reluDiscriminator/out/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
¨
Discriminator/out_1/BiasAddBiasAddDiscriminator/out_1/MatMulDiscriminator/out/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
Discriminator/out_1/SigmoidSigmoidDiscriminator/out_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
LogLogDiscriminator/out/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
`
subSubsub/xDiscriminator/out_1/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
C
Log_1Logsub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
H
addAddLogLog_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
A
NegNegadd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
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
 *  ?*
dtype0*
_output_shapes
: 
d
sub_1Subsub_1/xDiscriminator/out_1/Sigmoid*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
E
Log_2Logsub_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
generator_loss/tagConst*
dtype0*
_output_shapes
: *
valueB Bgenerator_loss
^
generator_lossHistogramSummarygenerator_loss/tagLog_2*
_output_shapes
: *
T0
R
gradients/ShapeShapeNeg*
T0*
out_type0*
_output_shapes
:
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients/Neg_grad/NegNeggradients/Fill*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/add_grad/SumSumgradients/Neg_grad/Neg(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ł
gradients/add_grad/Sum_1Sumgradients/Neg_grad/Neg*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ú
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
ŕ
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
gradients/Log_grad/Reciprocal
ReciprocalDiscriminator/out/Sigmoid,^gradients/add_grad/tuple/control_dependency*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Log_grad/mulMul+gradients/add_grad/tuple/control_dependencygradients/Log_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/Log_1_grad/Reciprocal
Reciprocalsub.^gradients/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
gradients/Log_1_grad/mulMul-gradients/add_grad/tuple/control_dependency_1gradients/Log_1_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
4gradients/Discriminator/out/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out/Sigmoidgradients/Log_grad/mul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
gradients/sub_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
u
gradients/sub_grad/Shape_1ShapeDiscriminator/out_1/Sigmoid*
_output_shapes
:*
T0*
out_type0
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ą
gradients/sub_grad/SumSumgradients/Log_1_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ľ
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

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
É
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
: 
ŕ
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ľ
4gradients/Discriminator/out/BiasAdd_grad/BiasAddGradBiasAddGrad4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
Ż
9gradients/Discriminator/out/BiasAdd_grad/tuple/group_depsNoOp5^gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad5^gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad
ş
Agradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyIdentity4gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/Discriminator/out/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
Cgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad:^gradients/Discriminator/out/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad
Ă
6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid-gradients/sub_grad/tuple/control_dependency_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ó
.gradients/Discriminator/out/MatMul_grad/MatMulMatMulAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(

0gradients/Discriminator/out/MatMul_grad/MatMul_1MatMul3Discriminator/second/Discriminator/secondleaky_reluAgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
¤
8gradients/Discriminator/out/MatMul_grad/tuple/group_depsNoOp/^gradients/Discriminator/out/MatMul_grad/MatMul1^gradients/Discriminator/out/MatMul_grad/MatMul_1
­
@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyIdentity.gradients/Discriminator/out/MatMul_grad/MatMul9^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*A
_class7
53loc:@gradients/Discriminator/out/MatMul_grad/MatMul
Ş
Bgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Identity0gradients/Discriminator/out/MatMul_grad/MatMul_19^gradients/Discriminator/out/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1
š
6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
_output_shapes
:*
T0*
data_formatNHWC
ľ
;gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp7^gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad7^gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
Â
Cgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity6gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*I
_class?
=;loc:@gradients/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
ˇ
Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity6gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad<^gradients/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*I
_class?
=;loc:@gradients/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
ż
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeShape7Discriminator/second/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Ę
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ę
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Shape@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*
_output_shapes
:*
T0*
out_type0

Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ą
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zerosFillJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_2Ngradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros/Const*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ogradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/second/Discriminator/secondleaky_relu/mul@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Xgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ShapeJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ă
Igradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectSelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqual@gradients/Discriminator/out/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ĺ
Kgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1SelectOgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/GreaterEqualHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/zeros@gradients/Discriminator/out/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
Fgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumSumIgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SelectXgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¨
Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/SumHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
Hgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1SumKgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Select_1Zgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ž
Lgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Sum_1Jgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
÷
Sgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeM^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1

[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/ReshapeT^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1T^gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
÷
0gradients/Discriminator/out_1/MatMul_grad/MatMulMatMulCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(

2gradients/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluCgradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
Ş
:gradients/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp1^gradients/Discriminator/out_1/MatMul_grad/MatMul3^gradients/Discriminator/out_1/MatMul_grad/MatMul_1
ľ
Bgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity0gradients/Discriminator/out_1/MatMul_grad/MatMul;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity2gradients/Discriminator/out_1/MatMul_grad/MatMul_1;^gradients/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	

gradients/AddNAddNCgradients/Discriminator/out/BiasAdd_grad/tuple/control_dependency_1Egradients/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1*
T0*G
_class=
;9loc:@gradients/Discriminator/out/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:

Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Î
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1Shape@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Đ
\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ShapeNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ł
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/MulMul[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency@Discriminator/second/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
Jgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumSumJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul\gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
˘
Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0

Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul9Discriminator/second/Discriminator/secondleaky_relu/alpha[gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Lgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Mul_1^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ş
Pgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Wgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1

_gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*a
_classW
USloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
ą
agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
_output_shapes
:*
T0*
out_type0
Î
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Î
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ˇ
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0

Qgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ę
Zgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ë
Kgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependencyJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
í
Mgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectQgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosBgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¸
Hgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumKgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectZgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ž
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeHgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
Jgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumMgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1\gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
´
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeJgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ý
Ugradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpM^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeO^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
Ł
]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityLgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeV^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*_
_classU
SQloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape
Š
_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1V^gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/AddN_1AddNBgradients/Discriminator/out/MatMul_grad/tuple/control_dependency_1Dgradients/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	*
T0*C
_class9
75loc:@gradients/Discriminator/out/MatMul_grad/MatMul_1*
N
÷
gradients/AddN_2AddN]gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/second/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_2*
T0*
data_formatNHWC*
_output_shapes	
:
Ů
`gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_2\^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad
ý
hgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_2a^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/second/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
jgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad

Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ň
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ö
^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapePgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Š
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
Lgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
¨
Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeLgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
¤
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha]gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
Ngradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1`gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ŕ
Rgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapeNgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Pgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ygradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeS^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
Ą
agradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeZ^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 
š
cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1Z^gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
č
Ugradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ď
Wgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul1Discriminator/first/Discriminator/firstleaky_reluhgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

_gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
É
ggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
igradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

ý
gradients/AddN_3AddN_gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1cgradients/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_3*
T0*
data_formatNHWC*
_output_shapes	
:
Ý
bgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_3^^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad

jgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_3c^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*a
_classW
USloc:@gradients/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
Ô
lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradc^gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ť
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeShape5Discriminator/first/Discriminator/firstleaky_relu/mul*
out_type0*
_output_shapes
:*
T0
Ć
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
ď
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Shapeggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ť
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosFillHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_2Lgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Mgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual5Discriminator/first/Discriminator/firstleaky_relu/mul>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
Vgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ShapeHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Ggradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectSelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Igradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1SelectMgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/GreaterEqualFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/zerosggradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
Dgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumSumGgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SelectVgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
˘
Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeReshapeDgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/SumFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Fgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1SumIgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Select_1Xgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
¨
Jgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeFgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Sum_1Hgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Shape_1*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
Qgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpI^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeK^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1

Ygradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityHgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/ReshapeR^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*[
_classQ
OMloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityJgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1R^gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
Wgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMuljgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ő
Ygradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relujgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

agradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulZ^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
Ń
igradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulb^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*j
_class`
^\loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ď
kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1b^gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:


gradients/AddN_4AddNjgradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1lgradients/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*n
_classd
b`loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:

Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ę
Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
Ę
Zgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ShapeLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulMulYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency>Discriminator/first/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ľ
Hgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumSumHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/MulZgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeHgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul7Discriminator/first/Discriminator/firstleaky_relu/alphaYgradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
Jgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1SumJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Mul_1\gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
´
Ngradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeJgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Sum_1Lgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ý
Ugradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpM^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeO^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1

]gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityLgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/ReshapeV^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*_
_classU
SQloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape
Š
_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityNgradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1V^gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ę
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
_output_shapes
:*
T0*
out_type0
ó
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapeigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ą
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ogradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Xgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Igradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Kgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectOgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosigradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Fgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumIgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectXgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
¨
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeFgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
¸
Hgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumKgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1Zgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ž
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeHgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
÷
Sgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpK^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeM^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1

[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityJgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeT^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityLgradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1T^gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients/AddN_5AddNigradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1kgradients/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*j
_class`
^\loc:@gradients/Discriminator/second/Discriminator/secondfully_connected/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
*
T0
ń
gradients/AddN_6AddN[gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1_gradients/Discriminator/first/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ˇ
Ygradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
_output_shapes	
:*
T0*
data_formatNHWC
Ő
^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6Z^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
÷
fgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/Discriminator/first/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
hgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityYgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad_^gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad

Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Î
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Đ
\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ł
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
Jgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul\gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˘
Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeJgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha[gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
Lgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ş
Pgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeLgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Ngradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Wgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpO^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeQ^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1

_gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityNgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeX^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: 
ą
agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityPgradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1X^gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
Sgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Ź
Ugradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulDiscriminator/realfgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

]gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpT^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMulV^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
Á
egradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentitySgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*f
_class\
ZXloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul
ż
ggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityUgradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1^^gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:

÷
gradients/AddN_7AddN]gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1agradients/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
data_formatNHWC*
_output_shapes	
:*
T0
Ů
`gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_7\^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad
ý
hgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_7a^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity[gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrada^gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*n
_classd
b`loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ć
Ugradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMulhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ź
Wgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhhgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

_gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpV^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulX^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
É
ggradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityUgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityWgradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1`^gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*j
_class`
^\loc:@gradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1

gradients/AddN_8AddNhgradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1jgradients/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*l
_classb
`^loc:@gradients/Discriminator/first/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes	
:*
T0

gradients/AddN_9AddNggradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1igradients/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
N* 
_output_shapes
:
*
T0*h
_class^
\Zloc:@gradients/Discriminator/first/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
Ž
beta1_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
ż
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
Ţ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 

beta1_power/readIdentitybeta1_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
Ž
beta2_power/initial_valueConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB
 *wž?*
dtype0*
_output_shapes
: 
ż
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
Ţ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(

beta2_power/readIdentitybeta2_power*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 

dDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ń
ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
­
TDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zerosFilldDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorZDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0

BDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
	container *
shape:


IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamTDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:


GDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:


fDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB"     *
dtype0*
_output_shapes
:
ó
\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ł
VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zerosFillfDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor\Discriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*

index_type0* 
_output_shapes
:


DDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel

KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignAssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
validate_shape(* 
_output_shapes
:


IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/readIdentityDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel* 
_output_shapes
:

ń
RDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ţ
@Discriminator/first/Discriminator/firstfully_connected/bias/Adam
VariableV2*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:

GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignAssign@Discriminator/first/Discriminator/firstfully_connected/bias/AdamRDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias

EDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/readIdentity@Discriminator/first/Discriminator/firstfully_connected/bias/Adam*
_output_shapes	
:*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
ó
TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:

BDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
	container *
shape:

IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignAssignBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes	
:

GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/readIdentityBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes	
:

fDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ő
\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ľ
VDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zerosFillfDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensor\Discriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:


DDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam
VariableV2*
shared_name *R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:


KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/kernel/AdamVDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:


IDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel* 
_output_shapes
:


hDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
÷
^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ť
XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zerosFillhDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensor^Discriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*

index_type0

FDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1
VariableV2*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Ą
MDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/AssignAssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:


KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/readIdentityFDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1* 
_output_shapes
:
*
T0*R
_classH
FDloc:@Discriminator/second/Discriminator/secondfully_connected/kernel
ő
TDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:

BDiscriminator/second/Discriminator/secondfully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
	container *
shape:

IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/AssignAssignBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamTDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(

GDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/readIdentityBDiscriminator/second/Discriminator/secondfully_connected/bias/Adam*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:
÷
VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:

DDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias

KDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/AssignAssignDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:

IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/readIdentityDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1*
T0*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
_output_shapes	
:
ł
/Discriminator/out/kernel/Adam/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Ŕ
Discriminator/out/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	
ţ
$Discriminator/out/kernel/Adam/AssignAssignDiscriminator/out/kernel/Adam/Discriminator/out/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel
¤
"Discriminator/out/kernel/Adam/readIdentityDiscriminator/out/kernel/Adam*
_output_shapes
:	*
T0*+
_class!
loc:@Discriminator/out/kernel
ľ
1Discriminator/out/kernel/Adam_1/Initializer/zerosConst*+
_class!
loc:@Discriminator/out/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
Â
Discriminator/out/kernel/Adam_1
VariableV2*
shared_name *+
_class!
loc:@Discriminator/out/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	

&Discriminator/out/kernel/Adam_1/AssignAssignDiscriminator/out/kernel/Adam_11Discriminator/out/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@Discriminator/out/kernel*
validate_shape(*
_output_shapes
:	
¨
$Discriminator/out/kernel/Adam_1/readIdentityDiscriminator/out/kernel/Adam_1*
T0*+
_class!
loc:@Discriminator/out/kernel*
_output_shapes
:	
Ľ
-Discriminator/out/bias/Adam/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
˛
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
ń
"Discriminator/out/bias/Adam/AssignAssignDiscriminator/out/bias/Adam-Discriminator/out/bias/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:

 Discriminator/out/bias/Adam/readIdentityDiscriminator/out/bias/Adam*
_output_shapes
:*
T0*)
_class
loc:@Discriminator/out/bias
§
/Discriminator/out/bias/Adam_1/Initializer/zerosConst*)
_class
loc:@Discriminator/out/bias*
valueB*    *
dtype0*
_output_shapes
:
´
Discriminator/out/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@Discriminator/out/bias*
	container 
÷
$Discriminator/out/bias/Adam_1/AssignAssignDiscriminator/out/bias/Adam_1/Discriminator/out/bias/Adam_1/Initializer/zeros*)
_class
loc:@Discriminator/out/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0

"Discriminator/out/bias/Adam_1/readIdentityDiscriminator/out/bias/Adam_1*
T0*)
_class
loc:@Discriminator/out/bias*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *ˇQ9
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
˝
SAdam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam	ApplyAdam=Discriminator/first/Discriminator/firstfully_connected/kernelBDiscriminator/first/Discriminator/firstfully_connected/kernel/AdamDDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
T0*P
_classF
DBloc:@Discriminator/first/Discriminator/firstfully_connected/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( 
Ž
QAdam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdam	ApplyAdam;Discriminator/first/Discriminator/firstfully_connected/bias@Discriminator/first/Discriminator/firstfully_connected/bias/AdamBDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_8*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
Ç
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

¸
SAdam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdam	ApplyAdam=Discriminator/second/Discriminator/secondfully_connected/biasBDiscriminator/second/Discriminator/secondfully_connected/bias/AdamDDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*P
_classF
DBloc:@Discriminator/second/Discriminator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0

.Adam/update_Discriminator/out/kernel/ApplyAdam	ApplyAdamDiscriminator/out/kernelDiscriminator/out/kernel/AdamDiscriminator/out/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
_output_shapes
:	*
use_locking( *
T0*+
_class!
loc:@Discriminator/out/kernel*
use_nesterov( 
ň
,Adam/update_Discriminator/out/bias/ApplyAdam	ApplyAdamDiscriminator/out/biasDiscriminator/out/bias/AdamDiscriminator/out/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*)
_class
loc:@Discriminator/out/bias
Ö
Adam/mulMulbeta1_power/read
Adam/beta1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
_output_shapes
: 
Ć
Adam/AssignAssignbeta1_powerAdam/mul*
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
Ř

Adam/mul_1Mulbeta2_power/read
Adam/beta2R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias
Ę
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*N
_classD
B@loc:@Discriminator/first/Discriminator/firstfully_connected/bias*
validate_shape(*
_output_shapes
: 
â
AdamNoOp^Adam/Assign^Adam/Assign_1R^Adam/update_Discriminator/first/Discriminator/firstfully_connected/bias/ApplyAdamT^Adam/update_Discriminator/first/Discriminator/firstfully_connected/kernel/ApplyAdam-^Adam/update_Discriminator/out/bias/ApplyAdam/^Adam/update_Discriminator/out/kernel/ApplyAdamT^Adam/update_Discriminator/second/Discriminator/secondfully_connected/bias/ApplyAdamV^Adam/update_Discriminator/second/Discriminator/secondfully_connected/kernel/ApplyAdam
V
gradients_1/ShapeShapeLog_2*
T0*
out_type0*
_output_shapes
:
Z
gradients_1/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 

gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
{
!gradients_1/Log_2_grad/Reciprocal
Reciprocalsub_1^gradients_1/Fill*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients_1/Log_2_grad/mulMulgradients_1/Fill!gradients_1/Log_2_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
Ŕ
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ť
gradients_1/sub_1_grad/SumSumgradients_1/Log_2_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ż
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
§
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
Ů
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape
đ
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGradSigmoidGradDiscriminator/out_1/Sigmoid1gradients_1/sub_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGradBiasAddGrad8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*
T0*
data_formatNHWC*
_output_shapes
:
ť
=gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_depsNoOp9^gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad9^gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad
Ę
Egradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyIdentity8gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/Sigmoid_grad/SigmoidGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
Ggradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency_1Identity8gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad>^gradients_1/Discriminator/out_1/BiasAdd_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/Discriminator/out_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ű
2gradients_1/Discriminator/out_1/MatMul_grad/MatMulMatMulEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependencyDiscriminator/out/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0

4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1MatMul5Discriminator/second_1/Discriminator/secondleaky_reluEgradients_1/Discriminator/out_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
°
<gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_depsNoOp3^gradients_1/Discriminator/out_1/MatMul_grad/MatMul5^gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1
˝
Dgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/Discriminator/out_1/MatMul_grad/MatMul=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*E
_class;
97loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul
ş
Fgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1=^gradients_1/Discriminator/out_1/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Discriminator/out_1/MatMul_grad/MatMul_1*
_output_shapes
:	
Ĺ
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeShape9Discriminator/second_1/Discriminator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Đ
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ň
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2ShapeDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
˝
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosFillNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_2Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Sgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualGreaterEqual9Discriminator/second_1/Discriminator/secondleaky_relu/mulBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Đ
\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ShapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ó
Mgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SelectSelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependencyLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
Ogradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1SelectSgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/GreaterEqualLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/zerosDgradients_1/Discriminator/out_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ž
Jgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumSumMgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select\gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
´
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeReshapeJgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/SumLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ä
Lgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1SumOgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Select_1^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
ş
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1ReshapeLgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Sum_1Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Wgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_depsNoOpO^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeQ^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
Ť
_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyIdentityNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/ReshapeX^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*a
_classW
USloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape
ą
agradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1IdentityPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1X^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Ô
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1ShapeBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ü
`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ShapeRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
­
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/MulMul_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependencyBDiscriminator/second_1/Discriminator/secondfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
Ngradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumSumNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul`gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ž
Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeReshapeNgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape*
Tshape0*
_output_shapes
: *
T0
¨
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1Mul;Discriminator/second_1/Discriminator/secondleaky_relu/alpha_gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
Pgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1SumPgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Mul_1bgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ć
Tgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1ReshapePgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Sum_1Rgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

[gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_depsNoOpS^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/ReshapeU^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1
Š
cgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityRgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*
_output_shapes
: *
T0*e
_class[
YWloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape
Á
egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1\^gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

gradients_1/AddNAddNagradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/tuple/control_dependency_1egradients_1/Discriminator/second_1/Discriminator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1
˝
_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
data_formatNHWC*
_output_shapes	
:*
T0
á
dgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN`^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad

lgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddNe^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*c
_classY
WUloc:@gradients_1/Discriminator/second_1/Discriminator/secondleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
ngradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity_gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrade^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
đ
Ygradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMulMatMullgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyDDiscriminator/second/Discriminator/secondfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ů
[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1MatMul3Discriminator/first_1/Discriminator/firstleaky_relulgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0
Ľ
cgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpZ^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul\^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
Ů
kgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityYgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMuld^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
mgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency_1Identity[gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1d^gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*n
_classd
b`loc:@gradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/MatMul_1
Á
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeShape7Discriminator/first_1/Discriminator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ě
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
÷
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Shapekgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ˇ
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zerosFillLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_2Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Qgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualGreaterEqual7Discriminator/first_1/Discriminator/firstleaky_relu/mul@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
Zgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ShapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Kgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectSelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualkgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependencyJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Mgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1SelectQgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/GreaterEqualJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/zeroskgradients_1/Discriminator/second_1/Discriminator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
Hgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumSumKgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SelectZgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ž
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeReshapeHgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/SumJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ž
Jgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1SumMgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Select_1\gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
´
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1ReshapeJgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Sum_1Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ý
Ugradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_depsNoOpM^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeO^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1
Ł
]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependencyIdentityLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/ReshapeV^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1IdentityNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1V^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Đ
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1Shape@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ö
^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ShapePgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
§
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/MulMul]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency@Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Á
Lgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumSumLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
¨
Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeReshapeLgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
˘
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1Mul9Discriminator/first_1/Discriminator/firstleaky_relu/alpha]gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
Ngradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1SumNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Mul_1`gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ŕ
Rgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1ReshapeNgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Sum_1Pgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

Ygradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_depsNoOpQ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeS^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1
Ą
agradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityPgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/ReshapeZ^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*c
_classY
WUloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0
š
cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityRgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1Z^gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˙
gradients_1/AddN_1AddN_gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/tuple/control_dependency_1cgradients_1/Discriminator/first_1/Discriminator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC*
_output_shapes	
:
ß
bgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_1^^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad

jgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1c^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*a
_classW
USloc:@gradients_1/Discriminator/first_1/Discriminator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
lgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGradc^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ę
Wgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulMatMuljgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyBDiscriminator/first/Discriminator/firstfully_connected/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
°
Ygradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/Tanhjgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

agradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpX^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulZ^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
Ń
igradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityWgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMulb^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps*j
_class`
^\loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ď
kgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityYgradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1b^gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/group_deps* 
_output_shapes
:
*
T0*l
_classb
`^loc:@gradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/MatMul_1
â
(gradients_1/Generator/Tanh_grad/TanhGradTanhGradGenerator/Tanhigradients_1/Discriminator/first_1/Discriminator/firstfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/Generator/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ł
9gradients_1/Generator/dense/BiasAdd_grad/tuple/group_depsNoOp)^gradients_1/Generator/Tanh_grad/TanhGrad5^gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad
Ł
Agradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/Generator/Tanh_grad/TanhGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/Generator/Tanh_grad/TanhGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
Cgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad:^gradients_1/Generator/dense/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/Generator/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ń
.gradients_1/Generator/dense/MatMul_grad/MatMulMatMulAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependencyGenerator/dense/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
ů
0gradients_1/Generator/dense/MatMul_grad/MatMul_1MatMul)Generator/forth/Generator/forthleaky_reluAgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 
¤
8gradients_1/Generator/dense/MatMul_grad/tuple/group_depsNoOp/^gradients_1/Generator/dense/MatMul_grad/MatMul1^gradients_1/Generator/dense/MatMul_grad/MatMul_1
­
@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/Generator/dense/MatMul_grad/MatMul9^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/Generator/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
Bgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/Generator/dense/MatMul_grad/MatMul_19^gradients_1/Generator/dense/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/Generator/dense/MatMul_grad/MatMul_1* 
_output_shapes
:

­
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeShape-Generator/forth/Generator/forthleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Á
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
Â
Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Shape@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zerosFillBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_2Fgradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ú
Ggradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqualGreaterEqual-Generator/forth/Generator/forthleaky_relu/mul?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Pgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ShapeBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ë
Agradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectSelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Í
Cgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1SelectGgradients_1/Generator/forth/Generator/forthleaky_relu_grad/GreaterEqual@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/zeros@gradients_1/Generator/dense/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/SumSumAgradients_1/Generator/forth/Generator/forthleaky_relu_grad/SelectPgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeReshape>gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
 
@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1SumCgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Select_1Rgradients_1/Generator/forth/Generator/forthleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Dgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Sum_1Bgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
Kgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeE^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1
ű
Sgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/forth/Generator/forthleaky_relu_grad/ReshapeL^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ugradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1L^gradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ĺ
Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1Shape?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
¸
Tgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ShapeFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulMulSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency?Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Bgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumSumBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/MulTgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Mul/Generator/forth/Generator/forthleaky_relu/alphaSgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
Dgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˘
Hgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ë
Ogradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1
ů
Wgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape*
_output_shapes
: 

Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
gradients_1/AddN_2AddNUgradients_1/Generator/forth/Generator/forthleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/forth/Generator/forthleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/forth/Generator/forthleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Ł
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
î
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_2fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ň
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_2hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ë
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Shape_1*
Tshape0*
_output_shapes	
:*
T0
Ą
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1
Ó
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Ě
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/forth/Generator/forthfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ł
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
î
fgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
¸
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ů
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ň
Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/forth/Generator/forthfully_connected/BiasAddigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ß
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ë
Zgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ą
agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
Ó
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1
ě
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ş
_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg
×
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/add_1_grad/Reshape_1
¸
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:

Sgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
_output_shapes	
:*
T0*
data_formatNHWC
˘
Xgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad
Ň
`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
bgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/BiasAddGrad
Ť
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/forth/Generator/forthbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:
°
Vgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/forth/Generator/forthbatch_normalized/moving_mean/read*
T0*
_output_shapes	
:

agradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
ž
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:*
T0*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul
Ä
kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
_output_shapes	
:*
T0*i
_class_
][loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/Mul_1
Î
Mgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/forth/Generator/forthfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
ˇ
Ogradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1MatMul)Generator/third/Generator/thirdleaky_relu`gradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

Wgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1
Š
_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
agradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/group_deps*b
_classX
VTloc:@gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0

gradients_1/AddN_3AddNkgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
_output_shapes	
:*
T0*m
_classc
a_loc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N
Ď
Rgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_3:Generator/forth/Generator/forthbatch_normalized/gamma/read*
_output_shapes	
:*
T0
Ö
Tgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_3?Generator/forth/Generator/forthbatch_normalized/batchnorm/Rsqrt*
_output_shapes	
:*
T0

_gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1
ś
ggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:
ź
igradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:*
T0
­
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeShape-Generator/third/Generator/thirdleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Á
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
á
Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Shape_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zerosFillBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_2Fgradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
ú
Ggradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqualGreaterEqual-Generator/third/Generator/thirdleaky_relu/mul?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
Pgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ShapeBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ę
Agradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectSelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
Cgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1SelectGgradients_1/Generator/third/Generator/thirdleaky_relu_grad/GreaterEqual@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/zeros_gradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/SumSumAgradients_1/Generator/third/Generator/thirdleaky_relu_grad/SelectPgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeReshape>gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1SumCgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Select_1Rgradients_1/Generator/third/Generator/thirdleaky_relu_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Dgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Sum_1Bgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
Kgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeE^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1
ű
Sgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/third/Generator/thirdleaky_relu_grad/ReshapeL^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ugradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1L^gradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1

Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ĺ
Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1Shape?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
¸
Tgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ShapeFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulMulSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency?Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Bgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumSumBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/MulTgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Mul/Generator/third/Generator/thirdleaky_relu/alphaSgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
Dgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
˘
Hgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
Ogradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1
ů
Wgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape*
_output_shapes
: 

Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
gradients_1/AddN_4AddNUgradients_1/Generator/third/Generator/thirdleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/third/Generator/thirdleaky_relu/mul_grad/tuple/control_dependency_1*
T0*W
_classM
KIloc:@gradients_1/Generator/third/Generator/thirdleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ő
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeShape?Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1*
_output_shapes
:*
T0*
out_type0
Ł
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
î
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_4fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ň
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_4hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ë
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ą
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1
Ó
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Ě
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeShape6Generator/third/Generator/thirdfully_connected/BiasAdd*
out_type0*
_output_shapes
:*
T0
Ł
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
î
fgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ShapeXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
¸
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumSumTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mulfgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ň
Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul6Generator/third/Generator/thirdfully_connected/BiasAddigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1SumVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Mul_1hgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ë
Zgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Sum_1Xgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
Ą
agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOpY^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape[^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1
Ó
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityXgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshapeb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape
Ě
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1IdentityZgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
_output_shapes	
:*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1
ě
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/NegNegkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
Ş
_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpl^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1S^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg
×
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitykgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
¸
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/Neg*
_output_shapes	
:

Sgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes	
:
˘
Xgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_depsNoOpj^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyT^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad
Ň
`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependencyIdentityigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*k
_classa
_]loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape
Ź
bgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/BiasAddGrad
Ť
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulMuligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1=Generator/third/Generator/thirdbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:
°
Vgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1Muligradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1@Generator/third/Generator/thirdbatch_normalized/moving_mean/read*
T0*
_output_shapes	
:

agradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/MulW^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1
ž
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mulb^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:*
T0
Ä
kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1b^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
Î
Mgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/third/Generator/thirdfully_connected/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
š
Ogradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1MatMul+Generator/second/Generator/secondleaky_relu`gradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0

Wgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1
Š
_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
agradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:


gradients_1/AddN_5AddNkgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1kgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*m
_classc
a_loc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
Ď
Rgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_5:Generator/third/Generator/thirdbatch_normalized/gamma/read*
T0*
_output_shapes	
:
Ö
Tgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_5?Generator/third/Generator/thirdbatch_normalized/batchnorm/Rsqrt*
_output_shapes	
:*
T0

_gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpS^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/MulU^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1
ś
ggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityRgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*e
_class[
YWloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:
ź
igradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1`^gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:
ą
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeShape/Generator/second/Generator/secondleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
Ĺ
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
ă
Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Shape_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/zerosFillDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_2Hgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Igradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualGreaterEqual/Generator/second/Generator/secondleaky_relu/mulAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
Rgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients_1/Generator/second/Generator/secondleaky_relu_grad/ShapeDgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
đ
Cgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectSelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqual_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependencyBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ň
Egradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1SelectIgradients_1/Generator/second/Generator/secondleaky_relu_grad/GreaterEqualBgradients_1/Generator/second/Generator/secondleaky_relu_grad/zeros_gradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumSumCgradients_1/Generator/second/Generator/secondleaky_relu_grad/SelectRgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeReshape@gradients_1/Generator/second/Generator/secondleaky_relu_grad/SumBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ś
Bgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1SumEgradients_1/Generator/second/Generator/secondleaky_relu_grad/Select_1Tgradients_1/Generator/second/Generator/secondleaky_relu_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Fgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1ReshapeBgradients_1/Generator/second/Generator/secondleaky_relu_grad/Sum_1Dgradients_1/Generator/second/Generator/secondleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
Mgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_depsNoOpE^gradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeG^gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1

Ugradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyIdentityDgradients_1/Generator/second/Generator/secondleaky_relu_grad/ReshapeN^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Wgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1IdentityFgradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1N^gradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1

Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
É
Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1ShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*
T0*
out_type0*
_output_shapes
:
ž
Vgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ShapeHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulMulUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependencyAGenerator/second/Generator/secondbatch_normalized/batchnorm/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Š
Dgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumSumDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/MulVgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeReshapeDgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 

Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Mul1Generator/second/Generator/secondleaky_relu/alphaUgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
Fgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1SumFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Mul_1Xgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
¨
Jgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1ReshapeFgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Sum_1Hgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ń
Qgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_depsNoOpI^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeK^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1

Ygradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependencyIdentityHgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/ReshapeR^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape*
_output_shapes
: 

[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1IdentityJgradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1R^gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ç
gradients_1/AddN_6AddNWgradients_1/Generator/second/Generator/secondleaky_relu_grad/tuple/control_dependency_1[gradients_1/Generator/second/Generator/secondleaky_relu/mul_grad/tuple/control_dependency_1*
T0*Y
_classO
MKloc:@gradients_1/Generator/second/Generator/secondleaky_relu_grad/Reshape_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ů
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeShapeAGenerator/second/Generator/secondbatch_normalized/batchnorm/mul_1*
T0*
out_type0*
_output_shapes
:
Ľ
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ô
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumSumgradients_1/AddN_6hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ř
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Sumgradients_1/AddN_6jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ń
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0
§
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1
Ű
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Đ
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeShape8Generator/second/Generator/secondfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ľ
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ô
hgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ShapeZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ž
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumSumVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mulhgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ř
Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/ReshapeReshapeVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
š
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1Mul8Generator/second/Generator/secondfully_connected/BiasAddkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ĺ
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1SumXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Mul_1jgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ń
\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1ReshapeXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Sum_1Zgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:
§
cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_depsNoOp[^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape]^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1
Ű
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyIdentityZgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshaped^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ô
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1Identity\gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
_output_shapes	
:
đ
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/NegNegmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
°
agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_depsNoOpn^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1U^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg
ß
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependencyIdentitymgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/tuple/control_dependency_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/add_1_grad/Reshape_1*
_output_shapes	
:
Ŕ
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1IdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Negb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/group_deps*
_output_shapes	
:*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/Neg

Ugradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes	
:*
T0
¨
Zgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_depsNoOpl^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependencyV^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad
Ú
bgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitykgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*
T0*m
_classc
a_loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
dgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentityUgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad[^gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/group_deps*h
_class^
\Zloc:@gradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
ą
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulMulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1?Generator/second/Generator/secondbatch_normalized/batchnorm/mul*
T0*
_output_shapes	
:
ś
Xgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1Mulkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency_1BGenerator/second/Generator/secondbatch_normalized/moving_mean/read*
T0*
_output_shapes	
:

cgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_depsNoOpW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/MulY^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1
Ć
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependencyIdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Muld^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul*
_output_shapes	
:
Ě
mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1IdentityXgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1d^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/group_deps*
T0*k
_classa
_]loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/Mul_1*
_output_shapes	
:
Ô
Ogradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulMatMulbgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency<Generator/second/Generator/secondfully_connected/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
ť
Qgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1MatMul)Generator/first/Generator/firstleaky_relubgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

Ygradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_depsNoOpP^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulR^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1
ą
agradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependencyIdentityOgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMulZ^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*b
_classX
VTloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul
Ż
cgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1IdentityQgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1Z^gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@gradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:


gradients_1/AddN_7AddNmgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/tuple/control_dependency_1mgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_2_grad/tuple/control_dependency_1*
T0*o
_classe
caloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_1_grad/Reshape_1*
N*
_output_shapes	
:
Ó
Tgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulMulgradients_1/AddN_7<Generator/second/Generator/secondbatch_normalized/gamma/read*
T0*
_output_shapes	
:
Ú
Vgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1Mulgradients_1/AddN_7AGenerator/second/Generator/secondbatch_normalized/batchnorm/Rsqrt*
_output_shapes	
:*
T0

agradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_depsNoOpU^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/MulW^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1
ž
igradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependencyIdentityTgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mulb^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*g
_class]
[Yloc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul*
_output_shapes	
:
Ä
kgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1IdentityVgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1b^gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/group_deps*
T0*i
_class_
][loc:@gradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/Mul_1*
_output_shapes	
:
­
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeShape-Generator/first/Generator/firstleaky_relu/mul*
T0*
out_type0*
_output_shapes
:
¸
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
ă
Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Shapeagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*
out_type0*
_output_shapes
:

Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0

@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosFillBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_2Fgradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
Ggradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualGreaterEqual-Generator/first/Generator/firstleaky_relu/mul6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
Pgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients_1/Generator/first/Generator/firstleaky_relu_grad/ShapeBgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ě
Agradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectSelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqualagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zeros*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
Cgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1SelectGgradients_1/Generator/first/Generator/firstleaky_relu_grad/GreaterEqual@gradients_1/Generator/first/Generator/firstleaky_relu_grad/zerosagradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

>gradients_1/Generator/first/Generator/firstleaky_relu_grad/SumSumAgradients_1/Generator/first/Generator/firstleaky_relu_grad/SelectPgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeReshape>gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
 
@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1SumCgradients_1/Generator/first/Generator/firstleaky_relu_grad/Select_1Rgradients_1/Generator/first/Generator/firstleaky_relu_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

Dgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1Reshape@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Sum_1Bgradients_1/Generator/first/Generator/firstleaky_relu_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ß
Kgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_depsNoOpC^gradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeE^gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
ű
Sgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependencyIdentityBgradients_1/Generator/first/Generator/firstleaky_relu_grad/ReshapeL^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*U
_classK
IGloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ugradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1IdentityDgradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1L^gradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
ź
Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1Shape6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*
out_type0*
_output_shapes
:
¸
Tgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgsBroadcastGradientArgsDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ShapeFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulMulSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency6Generator/first/Generator/firstfully_connected/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Bgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumSumBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/MulTgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeReshapeBgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape*
_output_shapes
: *
T0*
Tshape0

Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Mul/Generator/first/Generator/firstleaky_relu/alphaSgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
Dgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1SumDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Mul_1Vgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
˘
Hgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1ReshapeDgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Sum_1Fgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
Ogradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_depsNoOpG^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeI^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1
ů
Wgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependencyIdentityFgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/ReshapeP^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*Y
_classO
MKloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape*
_output_shapes
: *
T0

Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1IdentityHgradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1P^gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
á
gradients_1/AddN_8AddNUgradients_1/Generator/first/Generator/firstleaky_relu_grad/tuple/control_dependency_1Ygradients_1/Generator/first/Generator/firstleaky_relu/mul_grad/tuple/control_dependency_1*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1
ł
Sgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
T0*
data_formatNHWC*
_output_shapes	
:
Ë
Xgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_8T^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad
ç
`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8Y^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/Generator/first/Generator/firstleaky_relu_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
bgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1IdentitySgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGradY^gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*f
_class\
ZXloc:@gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/BiasAddGrad
Í
Mgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulMatMul`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency:Generator/first/Generator/firstfully_connected/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d*
transpose_a( *
transpose_b(

Ogradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1MatMulGenerator/noise`gradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	d*
transpose_a(*
transpose_b( 

Wgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_depsNoOpN^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulP^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1
¨
_gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependencyIdentityMgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMulX^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
Ś
agradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1IdentityOgradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1X^gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@gradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/MatMul_1*
_output_shapes
:	d

beta1_power_1/initial_valueConst*'
_class
loc:@Generator/dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 

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
˝
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
w
beta1_power_1/readIdentitybeta1_power_1*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 

beta2_power_1/initial_valueConst*
dtype0*
_output_shapes
: *'
_class
loc:@Generator/dense/bias*
valueB
 *wž?

beta2_power_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@Generator/dense/bias
˝
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
÷
\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
á
RGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    

LGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zerosFill\Generator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0*
_output_shapes
:	d
ú
:Generator/first/Generator/firstfully_connected/kernel/Adam
VariableV2*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name 
ň
AGenerator/first/Generator/firstfully_connected/kernel/Adam/AssignAssign:Generator/first/Generator/firstfully_connected/kernel/AdamLGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
validate_shape(*
_output_shapes
:	d
ű
?Generator/first/Generator/firstfully_connected/kernel/Adam/readIdentity:Generator/first/Generator/firstfully_connected/kernel/Adam*
_output_shapes
:	d*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
ů
^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB"d      *
dtype0*
_output_shapes
:
ă
TGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	d*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*

index_type0
ü
<Generator/first/Generator/firstfully_connected/kernel/Adam_1
VariableV2*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
	container *
shape:	d*
dtype0*
_output_shapes
:	d*
shared_name 
ř
CGenerator/first/Generator/firstfully_connected/kernel/Adam_1/AssignAssign<Generator/first/Generator/firstfully_connected/kernel/Adam_1NGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:	d*
use_locking(*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
˙
AGenerator/first/Generator/firstfully_connected/kernel/Adam_1/readIdentity<Generator/first/Generator/firstfully_connected/kernel/Adam_1*
_output_shapes
:	d*
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel
á
JGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
î
8Generator/first/Generator/firstfully_connected/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:
ć
?Generator/first/Generator/firstfully_connected/bias/Adam/AssignAssign8Generator/first/Generator/firstfully_connected/bias/AdamJGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(
ń
=Generator/first/Generator/firstfully_connected/bias/Adam/readIdentity8Generator/first/Generator/firstfully_connected/bias/Adam*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
ă
LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
đ
:Generator/first/Generator/firstfully_connected/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
	container *
shape:*
dtype0
ě
AGenerator/first/Generator/firstfully_connected/bias/Adam_1/AssignAssign:Generator/first/Generator/firstfully_connected/bias/Adam_1LGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias*
validate_shape(
ő
?Generator/first/Generator/firstfully_connected/bias/Adam_1/readIdentity:Generator/first/Generator/firstfully_connected/bias/Adam_1*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias
ű
^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"      
ĺ
TGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/ConstConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

NGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zerosFill^Generator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorTGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:


<Generator/second/Generator/secondfully_connected/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container 
ű
CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignAssign<Generator/second/Generator/secondfully_connected/kernel/AdamNGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

AGenerator/second/Generator/secondfully_connected/kernel/Adam/readIdentity<Generator/second/Generator/secondfully_connected/kernel/Adam*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:

ý
`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ç
VGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zerosFill`Generator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorVGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros/Const*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*

index_type0* 
_output_shapes
:
*
T0

>Generator/second/Generator/secondfully_connected/kernel/Adam_1
VariableV2*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:


EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignAssign>Generator/second/Generator/secondfully_connected/kernel/Adam_1PGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
validate_shape(* 
_output_shapes
:


CGenerator/second/Generator/secondfully_connected/kernel/Adam_1/readIdentity>Generator/second/Generator/secondfully_connected/kernel/Adam_1*
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel* 
_output_shapes
:

ĺ
LGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ň
:Generator/second/Generator/secondfully_connected/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
î
AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAssign:Generator/second/Generator/secondfully_connected/bias/AdamLGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:
÷
?Generator/second/Generator/secondfully_connected/bias/Adam/readIdentity:Generator/second/Generator/secondfully_connected/bias/Adam*
_output_shapes	
:*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
ç
NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
ô
<Generator/second/Generator/secondfully_connected/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
	container *
shape:*
dtype0
ô
CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignAssign<Generator/second/Generator/secondfully_connected/bias/Adam_1NGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
validate_shape(*
_output_shapes	
:
ű
AGenerator/second/Generator/secondfully_connected/bias/Adam_1/readIdentity<Generator/second/Generator/secondfully_connected/bias/Adam_1*
_output_shapes	
:*
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias
é
NGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ö
<Generator/second/Generator/secondbatch_normalized/gamma/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container 
ö
CGenerator/second/Generator/secondbatch_normalized/gamma/Adam/AssignAssign<Generator/second/Generator/secondbatch_normalized/gamma/AdamNGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zeros*
use_locking(*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:
ý
AGenerator/second/Generator/secondbatch_normalized/gamma/Adam/readIdentity<Generator/second/Generator/secondbatch_normalized/gamma/Adam*
_output_shapes	
:*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma
ë
PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zerosConst*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ř
>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
	container 
ü
EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignAssign>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1PGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(

CGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/readIdentity>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1*
_output_shapes	
:*
T0*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma
ç
MGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB*    *
dtype0*
_output_shapes	
:
ô
;Generator/second/Generator/secondbatch_normalized/beta/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
	container *
shape:
ň
BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignAssign;Generator/second/Generator/secondbatch_normalized/beta/AdamMGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ú
@Generator/second/Generator/secondbatch_normalized/beta/Adam/readIdentity;Generator/second/Generator/secondbatch_normalized/beta/Adam*
_output_shapes	
:*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
é
OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zerosConst*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
valueB*    *
dtype0*
_output_shapes	
:
ö
=Generator/second/Generator/secondbatch_normalized/beta/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta
ř
DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignAssign=Generator/second/Generator/secondbatch_normalized/beta/Adam_1OGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ţ
BGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/readIdentity=Generator/second/Generator/secondbatch_normalized/beta/Adam_1*
T0*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
_output_shapes	
:
÷
\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
á
RGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0

LGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zerosFill\Generator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0* 
_output_shapes
:

ü
:Generator/third/Generator/thirdfully_connected/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:

ó
AGenerator/third/Generator/thirdfully_connected/kernel/Adam/AssignAssign:Generator/third/Generator/thirdfully_connected/kernel/AdamLGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ü
?Generator/third/Generator/thirdfully_connected/kernel/Adam/readIdentity:Generator/third/Generator/thirdfully_connected/kernel/Adam* 
_output_shapes
:
*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel
ů
^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
ă
TGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*

index_type0* 
_output_shapes
:

ţ
<Generator/third/Generator/thirdfully_connected/kernel/Adam_1
VariableV2*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ů
CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignAssign<Generator/third/Generator/thirdfully_connected/kernel/Adam_1NGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
validate_shape(* 
_output_shapes
:


AGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/readIdentity<Generator/third/Generator/thirdfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel* 
_output_shapes
:

á
JGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
î
8Generator/third/Generator/thirdfully_connected/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container 
ć
?Generator/third/Generator/thirdfully_connected/bias/Adam/AssignAssign8Generator/third/Generator/thirdfully_connected/bias/AdamJGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(
ń
=Generator/third/Generator/thirdfully_connected/bias/Adam/readIdentity8Generator/third/Generator/thirdfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:
ă
LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zerosConst*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
valueB*    *
dtype0*
_output_shapes	
:
đ
:Generator/third/Generator/thirdfully_connected/bias/Adam_1
VariableV2*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ě
AGenerator/third/Generator/thirdfully_connected/bias/Adam_1/AssignAssign:Generator/third/Generator/thirdfully_connected/bias/Adam_1LGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ő
?Generator/third/Generator/thirdfully_connected/bias/Adam_1/readIdentity:Generator/third/Generator/thirdfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias*
_output_shapes	
:
ĺ
LGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB*    
ň
:Generator/third/Generator/thirdbatch_normalized/gamma/Adam
VariableV2*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
î
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAssign:Generator/third/Generator/thirdbatch_normalized/gamma/AdamLGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(
÷
?Generator/third/Generator/thirdbatch_normalized/gamma/Adam/readIdentity:Generator/third/Generator/thirdbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:
ç
NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zerosConst*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
valueB*    *
dtype0*
_output_shapes	
:
ô
<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma
ô
CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignAssign<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
ű
AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/readIdentity<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
_output_shapes	
:*
T0
ă
KGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB*    *
dtype0*
_output_shapes	
:
đ
9Generator/third/Generator/thirdbatch_normalized/beta/Adam
VariableV2*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ę
@Generator/third/Generator/thirdbatch_normalized/beta/Adam/AssignAssign9Generator/third/Generator/thirdbatch_normalized/beta/AdamKGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zeros*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ô
>Generator/third/Generator/thirdbatch_normalized/beta/Adam/readIdentity9Generator/third/Generator/thirdbatch_normalized/beta/Adam*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:*
T0
ĺ
MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zerosConst*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
valueB*    *
dtype0*
_output_shapes	
:
ň
;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta
đ
BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignAssign;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1MGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ř
@Generator/third/Generator/thirdbatch_normalized/beta/Adam_1/readIdentity;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1*
T0*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
_output_shapes	
:
÷
\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0*
_output_shapes
:
á
RGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

LGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0* 
_output_shapes
:

ü
:Generator/forth/Generator/forthfully_connected/kernel/Adam
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel
ó
AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAssign:Generator/forth/Generator/forthfully_connected/kernel/AdamLGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros* 
_output_shapes
:
*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(
ü
?Generator/forth/Generator/forthfully_connected/kernel/Adam/readIdentity:Generator/forth/Generator/forthfully_connected/kernel/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:

ů
^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB"      *
dtype0
ă
TGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*

index_type0
ţ
<Generator/forth/Generator/forthfully_connected/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
	container *
shape:

ů
CGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/AssignAssign<Generator/forth/Generator/forthfully_connected/kernel/Adam_1NGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
validate_shape(* 
_output_shapes
:


AGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/readIdentity<Generator/forth/Generator/forthfully_connected/kernel/Adam_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel* 
_output_shapes
:

í
ZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
Ý
PGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 

JGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zerosFillZGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/shape_as_tensorPGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros/Const*
_output_shapes	
:*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0
î
8Generator/forth/Generator/forthfully_connected/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container 
ć
?Generator/forth/Generator/forthfully_connected/bias/Adam/AssignAssign8Generator/forth/Generator/forthfully_connected/bias/AdamJGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ń
=Generator/forth/Generator/forthfully_connected/bias/Adam/readIdentity8Generator/forth/Generator/forthfully_connected/bias/Adam*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:
ď
\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB:*
dtype0*
_output_shapes
:
ß
RGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/ConstConst*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
valueB
 *    *
dtype0*
_output_shapes
: 

LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zerosFill\Generator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros/Const*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*

index_type0*
_output_shapes	
:*
T0
đ
:Generator/forth/Generator/forthfully_connected/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
	container *
shape:*
dtype0
ě
AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAssign:Generator/forth/Generator/forthfully_connected/bias/Adam_1LGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
validate_shape(*
_output_shapes	
:
ő
?Generator/forth/Generator/forthfully_connected/bias/Adam_1/readIdentity:Generator/forth/Generator/forthfully_connected/bias/Adam_1*
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
_output_shapes	
:
ń
\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:
á
RGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 

LGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zerosFill\Generator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/shape_as_tensorRGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:
ň
:Generator/forth/Generator/forthbatch_normalized/gamma/Adam
VariableV2*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
î
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAssign:Generator/forth/Generator/forthbatch_normalized/gamma/AdamLGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
÷
?Generator/forth/Generator/forthbatch_normalized/gamma/Adam/readIdentity:Generator/forth/Generator/forthbatch_normalized/gamma/Adam*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:
ó
^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB:*
dtype0*
_output_shapes
:
ă
TGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/ConstConst*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
valueB
 *    *
dtype0*
_output_shapes
: 

NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zerosFill^Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/shape_as_tensorTGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros/Const*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*

index_type0*
_output_shapes	
:
ô
<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
	container 
ô
CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignAssign<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma
ű
AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/readIdentity<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
_output_shapes	
:
ď
[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:
ß
QGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 

KGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zerosFill[Generator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/shape_as_tensorQGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros/Const*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0*
_output_shapes	
:
đ
9Generator/forth/Generator/forthbatch_normalized/beta/Adam
VariableV2*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
ę
@Generator/forth/Generator/forthbatch_normalized/beta/Adam/AssignAssign9Generator/forth/Generator/forthbatch_normalized/beta/AdamKGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ô
>Generator/forth/Generator/forthbatch_normalized/beta/Adam/readIdentity9Generator/forth/Generator/forthbatch_normalized/beta/Adam*
_output_shapes	
:*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
ń
]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB:*
dtype0*
_output_shapes
:
á
SGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/ConstConst*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
valueB
 *    *
dtype0*
_output_shapes
: 

MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zerosFill]Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/shape_as_tensorSGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros/Const*
_output_shapes	
:*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*

index_type0
ň
;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
đ
BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignAssign;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1MGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
validate_shape(*
_output_shapes	
:
ř
@Generator/forth/Generator/forthbatch_normalized/beta/Adam_1/readIdentity;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1*
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta*
_output_shapes	
:
š
=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ł
3Generator/dense/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

-Generator/dense/kernel/Adam/Initializer/zerosFill=Generator/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3Generator/dense/kernel/Adam/Initializer/zeros/Const*)
_class
loc:@Generator/dense/kernel*

index_type0* 
_output_shapes
:
*
T0
ž
Generator/dense/kernel/Adam
VariableV2*)
_class
loc:@Generator/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
÷
"Generator/dense/kernel/Adam/AssignAssignGenerator/dense/kernel/Adam-Generator/dense/kernel/Adam/Initializer/zeros*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

 Generator/dense/kernel/Adam/readIdentityGenerator/dense/kernel/Adam*
T0*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:

ť
?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@Generator/dense/kernel*
valueB"     *
dtype0*
_output_shapes
:
Ľ
5Generator/dense/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@Generator/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 

/Generator/dense/kernel/Adam_1/Initializer/zerosFill?Generator/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5Generator/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@Generator/dense/kernel*

index_type0* 
_output_shapes
:

Ŕ
Generator/dense/kernel/Adam_1
VariableV2*
shared_name *)
_class
loc:@Generator/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$Generator/dense/kernel/Adam_1/AssignAssignGenerator/dense/kernel/Adam_1/Generator/dense/kernel/Adam_1/Initializer/zeros*)
_class
loc:@Generator/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ł
"Generator/dense/kernel/Adam_1/readIdentityGenerator/dense/kernel/Adam_1*)
_class
loc:@Generator/dense/kernel* 
_output_shapes
:
*
T0
Ł
+Generator/dense/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*'
_class
loc:@Generator/dense/bias*
valueB*    *
dtype0
°
Generator/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:
ę
 Generator/dense/bias/Adam/AssignAssignGenerator/dense/bias/Adam+Generator/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes	
:

Generator/dense/bias/Adam/readIdentityGenerator/dense/bias/Adam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes	
:
Ľ
-Generator/dense/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@Generator/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
˛
Generator/dense/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@Generator/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
đ
"Generator/dense/bias/Adam_1/AssignAssignGenerator/dense/bias/Adam_1-Generator/dense/bias/Adam_1/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(

 Generator/dense/bias/Adam_1/readIdentityGenerator/dense/bias/Adam_1*
T0*'
_class
loc:@Generator/dense/bias*
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
ó
MAdam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/first/Generator/firstfully_connected/kernel:Generator/first/Generator/firstfully_connected/kernel/Adam<Generator/first/Generator/firstfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/first/Generator/firstfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/first/Generator/firstfully_connected/kernel*
use_nesterov( *
_output_shapes
:	d
ć
KAdam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdam	ApplyAdam3Generator/first/Generator/firstfully_connected/bias8Generator/first/Generator/firstfully_connected/bias/Adam:Generator/first/Generator/firstfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/first/Generator/firstfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*F
_class<
:8loc:@Generator/first/Generator/firstfully_connected/bias

OAdam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdam	ApplyAdam7Generator/second/Generator/secondfully_connected/kernel<Generator/second/Generator/secondfully_connected/kernel/Adam>Generator/second/Generator/secondfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsiloncgradients_1/Generator/second/Generator/secondfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*J
_class@
><loc:@Generator/second/Generator/secondfully_connected/kernel*
use_nesterov( * 
_output_shapes
:

ň
MAdam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdam	ApplyAdam5Generator/second/Generator/secondfully_connected/bias:Generator/second/Generator/secondfully_connected/bias/Adam<Generator/second/Generator/secondfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilondgradients_1/Generator/second/Generator/secondfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/second/Generator/secondfully_connected/bias*
use_nesterov( *
_output_shapes	
:

OAdam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdam	ApplyAdam7Generator/second/Generator/secondbatch_normalized/gamma<Generator/second/Generator/secondbatch_normalized/gamma/Adam>Generator/second/Generator/secondbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonkgradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*J
_class@
><loc:@Generator/second/Generator/secondbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
ü
NAdam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdam	ApplyAdam6Generator/second/Generator/secondbatch_normalized/beta;Generator/second/Generator/secondbatch_normalized/beta/Adam=Generator/second/Generator/secondbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/second/Generator/secondbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*I
_class?
=;loc:@Generator/second/Generator/secondbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
ô
MAdam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdfully_connected/kernel:Generator/third/Generator/thirdfully_connected/kernel/Adam<Generator/third/Generator/thirdfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/third/Generator/thirdfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdfully_connected/kernel*
use_nesterov( * 
_output_shapes
:

ć
KAdam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdam	ApplyAdam3Generator/third/Generator/thirdfully_connected/bias8Generator/third/Generator/thirdfully_connected/bias/Adam:Generator/third/Generator/thirdfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/third/Generator/thirdfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*F
_class<
:8loc:@Generator/third/Generator/thirdfully_connected/bias
÷
MAdam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/third/Generator/thirdbatch_normalized/gamma:Generator/third/Generator/thirdbatch_normalized/gamma/Adam<Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*H
_class>
<:loc:@Generator/third/Generator/thirdbatch_normalized/gamma*
use_nesterov( 
đ
LAdam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/third/Generator/thirdbatch_normalized/beta9Generator/third/Generator/thirdbatch_normalized/beta/Adam;Generator/third/Generator/thirdbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/third/Generator/thirdbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*G
_class=
;9loc:@Generator/third/Generator/thirdbatch_normalized/beta*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0
ô
MAdam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthfully_connected/kernel:Generator/forth/Generator/forthfully_connected/kernel/Adam<Generator/forth/Generator/forthfully_connected/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonagradients_1/Generator/forth/Generator/forthfully_connected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*H
_class>
<:loc:@Generator/forth/Generator/forthfully_connected/kernel*
use_nesterov( * 
_output_shapes
:

ć
KAdam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdam	ApplyAdam3Generator/forth/Generator/forthfully_connected/bias8Generator/forth/Generator/forthfully_connected/bias/Adam:Generator/forth/Generator/forthfully_connected/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonbgradients_1/Generator/forth/Generator/forthfully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*F
_class<
:8loc:@Generator/forth/Generator/forthfully_connected/bias*
use_nesterov( *
_output_shapes	
:
÷
MAdam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdam	ApplyAdam5Generator/forth/Generator/forthbatch_normalized/gamma:Generator/forth/Generator/forthbatch_normalized/gamma/Adam<Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonigradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/mul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@Generator/forth/Generator/forthbatch_normalized/gamma*
use_nesterov( *
_output_shapes	
:*
use_locking( 
đ
LAdam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdam	ApplyAdam4Generator/forth/Generator/forthbatch_normalized/beta9Generator/forth/Generator/forthbatch_normalized/beta/Adam;Generator/forth/Generator/forthbatch_normalized/beta/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonggradients_1/Generator/forth/Generator/forthbatch_normalized/batchnorm/sub_grad/tuple/control_dependency*
use_nesterov( *
_output_shapes	
:*
use_locking( *
T0*G
_class=
;9loc:@Generator/forth/Generator/forthbatch_normalized/beta
ş
.Adam_1/update_Generator/dense/kernel/ApplyAdam	ApplyAdamGenerator/dense/kernelGenerator/dense/kernel/AdamGenerator/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/Generator/dense/MatMul_grad/tuple/control_dependency_1*
T0*)
_class
loc:@Generator/dense/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( 
Ź
,Adam_1/update_Generator/dense/bias/ApplyAdam	ApplyAdamGenerator/dense/biasGenerator/dense/bias/AdamGenerator/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/Generator/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
use_nesterov( 
ş


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
T0*'
_class
loc:@Generator/dense/bias*
_output_shapes
: 
Ľ
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
_output_shapes
: *
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(
ź

Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam*
_output_shapes
: *
T0*'
_class
loc:@Generator/dense/bias
Š
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*'
_class
loc:@Generator/dense/bias*
validate_shape(*
_output_shapes
: 
í	
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1-^Adam_1/update_Generator/dense/bias/ApplyAdam/^Adam_1/update_Generator/dense/kernel/ApplyAdamL^Adam_1/update_Generator/first/Generator/firstfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/first/Generator/firstfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/forth/Generator/forthfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/forth/Generator/forthfully_connected/kernel/ApplyAdamO^Adam_1/update_Generator/second/Generator/secondbatch_normalized/beta/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondbatch_normalized/gamma/ApplyAdamN^Adam_1/update_Generator/second/Generator/secondfully_connected/bias/ApplyAdamP^Adam_1/update_Generator/second/Generator/secondfully_connected/kernel/ApplyAdamM^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/beta/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdbatch_normalized/gamma/ApplyAdamL^Adam_1/update_Generator/third/Generator/thirdfully_connected/bias/ApplyAdamN^Adam_1/update_Generator/third/Generator/thirdfully_connected/kernel/ApplyAdam
g
Merge/MergeSummaryMergeSummarydiscriminator_lossgenerator_loss*
N*
_output_shapes
: ""ź
	variables­Š

7Generator/first/Generator/firstfully_connected/kernel:0<Generator/first/Generator/firstfully_connected/kernel/Assign<Generator/first/Generator/firstfully_connected/kernel/read:02RGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform:08
ú
5Generator/first/Generator/firstfully_connected/bias:0:Generator/first/Generator/firstfully_connected/bias/Assign:Generator/first/Generator/firstfully_connected/bias/read:02GGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros:08

9Generator/second/Generator/secondfully_connected/kernel:0>Generator/second/Generator/secondfully_connected/kernel/Assign>Generator/second/Generator/secondfully_connected/kernel/read:02TGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform:08

7Generator/second/Generator/secondfully_connected/bias:0<Generator/second/Generator/secondfully_connected/bias/Assign<Generator/second/Generator/secondfully_connected/bias/read:02IGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros:08

9Generator/second/Generator/secondbatch_normalized/gamma:0>Generator/second/Generator/secondbatch_normalized/gamma/Assign>Generator/second/Generator/secondbatch_normalized/gamma/read:02JGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones:08

8Generator/second/Generator/secondbatch_normalized/beta:0=Generator/second/Generator/secondbatch_normalized/beta/Assign=Generator/second/Generator/secondbatch_normalized/beta/read:02JGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros:08
 
?Generator/second/Generator/secondbatch_normalized/moving_mean:0DGenerator/second/Generator/secondbatch_normalized/moving_mean/AssignDGenerator/second/Generator/secondbatch_normalized/moving_mean/read:02QGenerator/second/Generator/secondbatch_normalized/moving_mean/Initializer/zeros:0
Ż
CGenerator/second/Generator/secondbatch_normalized/moving_variance:0HGenerator/second/Generator/secondbatch_normalized/moving_variance/AssignHGenerator/second/Generator/secondbatch_normalized/moving_variance/read:02TGenerator/second/Generator/secondbatch_normalized/moving_variance/Initializer/ones:0

7Generator/third/Generator/thirdfully_connected/kernel:0<Generator/third/Generator/thirdfully_connected/kernel/Assign<Generator/third/Generator/thirdfully_connected/kernel/read:02RGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform:08
ú
5Generator/third/Generator/thirdfully_connected/bias:0:Generator/third/Generator/thirdfully_connected/bias/Assign:Generator/third/Generator/thirdfully_connected/bias/read:02GGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros:08

7Generator/third/Generator/thirdbatch_normalized/gamma:0<Generator/third/Generator/thirdbatch_normalized/gamma/Assign<Generator/third/Generator/thirdbatch_normalized/gamma/read:02HGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones:08
ţ
6Generator/third/Generator/thirdbatch_normalized/beta:0;Generator/third/Generator/thirdbatch_normalized/beta/Assign;Generator/third/Generator/thirdbatch_normalized/beta/read:02HGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros:08

=Generator/third/Generator/thirdbatch_normalized/moving_mean:0BGenerator/third/Generator/thirdbatch_normalized/moving_mean/AssignBGenerator/third/Generator/thirdbatch_normalized/moving_mean/read:02OGenerator/third/Generator/thirdbatch_normalized/moving_mean/Initializer/zeros:0
§
AGenerator/third/Generator/thirdbatch_normalized/moving_variance:0FGenerator/third/Generator/thirdbatch_normalized/moving_variance/AssignFGenerator/third/Generator/thirdbatch_normalized/moving_variance/read:02RGenerator/third/Generator/thirdbatch_normalized/moving_variance/Initializer/ones:0

7Generator/forth/Generator/forthfully_connected/kernel:0<Generator/forth/Generator/forthfully_connected/kernel/Assign<Generator/forth/Generator/forthfully_connected/kernel/read:02RGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform:08
ú
5Generator/forth/Generator/forthfully_connected/bias:0:Generator/forth/Generator/forthfully_connected/bias/Assign:Generator/forth/Generator/forthfully_connected/bias/read:02GGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros:08

7Generator/forth/Generator/forthbatch_normalized/gamma:0<Generator/forth/Generator/forthbatch_normalized/gamma/Assign<Generator/forth/Generator/forthbatch_normalized/gamma/read:02HGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones:08
ţ
6Generator/forth/Generator/forthbatch_normalized/beta:0;Generator/forth/Generator/forthbatch_normalized/beta/Assign;Generator/forth/Generator/forthbatch_normalized/beta/read:02HGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros:08

=Generator/forth/Generator/forthbatch_normalized/moving_mean:0BGenerator/forth/Generator/forthbatch_normalized/moving_mean/AssignBGenerator/forth/Generator/forthbatch_normalized/moving_mean/read:02OGenerator/forth/Generator/forthbatch_normalized/moving_mean/Initializer/zeros:0
§
AGenerator/forth/Generator/forthbatch_normalized/moving_variance:0FGenerator/forth/Generator/forthbatch_normalized/moving_variance/AssignFGenerator/forth/Generator/forthbatch_normalized/moving_variance/read:02RGenerator/forth/Generator/forthbatch_normalized/moving_variance/Initializer/ones:0

Generator/dense/kernel:0Generator/dense/kernel/AssignGenerator/dense/kernel/read:023Generator/dense/kernel/Initializer/random_uniform:08
~
Generator/dense/bias:0Generator/dense/bias/AssignGenerator/dense/bias/read:02(Generator/dense/bias/Initializer/zeros:08
Ť
?Discriminator/first/Discriminator/firstfully_connected/kernel:0DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/read:02ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform:08

=Discriminator/first/Discriminator/firstfully_connected/bias:0BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignBDiscriminator/first/Discriminator/firstfully_connected/bias/read:02ODiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros:08
ł
ADiscriminator/second/Discriminator/secondfully_connected/kernel:0FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/read:02\Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform:08
˘
?Discriminator/second/Discriminator/secondfully_connected/bias:0DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignDDiscriminator/second/Discriminator/secondfully_connected/bias/read:02QDiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros:08

Discriminator/out/kernel:0Discriminator/out/kernel/AssignDiscriminator/out/kernel/read:025Discriminator/out/kernel/Initializer/random_uniform:08

Discriminator/out/bias:0Discriminator/out/bias/AssignDiscriminator/out/bias/read:02*Discriminator/out/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
´
DDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam:0IDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/AssignIDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/read:02VDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam/Initializer/zeros:0
ź
FDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1:0KDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/AssignKDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/read:02XDiscriminator/first/Discriminator/firstfully_connected/kernel/Adam_1/Initializer/zeros:0
Ź
BDiscriminator/first/Discriminator/firstfully_connected/bias/Adam:0GDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/AssignGDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/read:02TDiscriminator/first/Discriminator/firstfully_connected/bias/Adam/Initializer/zeros:0
´
DDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1:0IDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/AssignIDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/read:02VDiscriminator/first/Discriminator/firstfully_connected/bias/Adam_1/Initializer/zeros:0
ź
FDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam:0KDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/AssignKDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/read:02XDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam/Initializer/zeros:0
Ä
HDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1:0MDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/AssignMDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/read:02ZDiscriminator/second/Discriminator/secondfully_connected/kernel/Adam_1/Initializer/zeros:0
´
DDiscriminator/second/Discriminator/secondfully_connected/bias/Adam:0IDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/AssignIDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/read:02VDiscriminator/second/Discriminator/secondfully_connected/bias/Adam/Initializer/zeros:0
ź
FDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1:0KDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/AssignKDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/read:02XDiscriminator/second/Discriminator/secondfully_connected/bias/Adam_1/Initializer/zeros:0
 
Discriminator/out/kernel/Adam:0$Discriminator/out/kernel/Adam/Assign$Discriminator/out/kernel/Adam/read:021Discriminator/out/kernel/Adam/Initializer/zeros:0
¨
!Discriminator/out/kernel/Adam_1:0&Discriminator/out/kernel/Adam_1/Assign&Discriminator/out/kernel/Adam_1/read:023Discriminator/out/kernel/Adam_1/Initializer/zeros:0

Discriminator/out/bias/Adam:0"Discriminator/out/bias/Adam/Assign"Discriminator/out/bias/Adam/read:02/Discriminator/out/bias/Adam/Initializer/zeros:0
 
Discriminator/out/bias/Adam_1:0$Discriminator/out/bias/Adam_1/Assign$Discriminator/out/bias/Adam_1/read:021Discriminator/out/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0

<Generator/first/Generator/firstfully_connected/kernel/Adam:0AGenerator/first/Generator/firstfully_connected/kernel/Adam/AssignAGenerator/first/Generator/firstfully_connected/kernel/Adam/read:02NGenerator/first/Generator/firstfully_connected/kernel/Adam/Initializer/zeros:0

>Generator/first/Generator/firstfully_connected/kernel/Adam_1:0CGenerator/first/Generator/firstfully_connected/kernel/Adam_1/AssignCGenerator/first/Generator/firstfully_connected/kernel/Adam_1/read:02PGenerator/first/Generator/firstfully_connected/kernel/Adam_1/Initializer/zeros:0

:Generator/first/Generator/firstfully_connected/bias/Adam:0?Generator/first/Generator/firstfully_connected/bias/Adam/Assign?Generator/first/Generator/firstfully_connected/bias/Adam/read:02LGenerator/first/Generator/firstfully_connected/bias/Adam/Initializer/zeros:0

<Generator/first/Generator/firstfully_connected/bias/Adam_1:0AGenerator/first/Generator/firstfully_connected/bias/Adam_1/AssignAGenerator/first/Generator/firstfully_connected/bias/Adam_1/read:02NGenerator/first/Generator/firstfully_connected/bias/Adam_1/Initializer/zeros:0

>Generator/second/Generator/secondfully_connected/kernel/Adam:0CGenerator/second/Generator/secondfully_connected/kernel/Adam/AssignCGenerator/second/Generator/secondfully_connected/kernel/Adam/read:02PGenerator/second/Generator/secondfully_connected/kernel/Adam/Initializer/zeros:0
¤
@Generator/second/Generator/secondfully_connected/kernel/Adam_1:0EGenerator/second/Generator/secondfully_connected/kernel/Adam_1/AssignEGenerator/second/Generator/secondfully_connected/kernel/Adam_1/read:02RGenerator/second/Generator/secondfully_connected/kernel/Adam_1/Initializer/zeros:0

<Generator/second/Generator/secondfully_connected/bias/Adam:0AGenerator/second/Generator/secondfully_connected/bias/Adam/AssignAGenerator/second/Generator/secondfully_connected/bias/Adam/read:02NGenerator/second/Generator/secondfully_connected/bias/Adam/Initializer/zeros:0

>Generator/second/Generator/secondfully_connected/bias/Adam_1:0CGenerator/second/Generator/secondfully_connected/bias/Adam_1/AssignCGenerator/second/Generator/secondfully_connected/bias/Adam_1/read:02PGenerator/second/Generator/secondfully_connected/bias/Adam_1/Initializer/zeros:0

>Generator/second/Generator/secondbatch_normalized/gamma/Adam:0CGenerator/second/Generator/secondbatch_normalized/gamma/Adam/AssignCGenerator/second/Generator/secondbatch_normalized/gamma/Adam/read:02PGenerator/second/Generator/secondbatch_normalized/gamma/Adam/Initializer/zeros:0
¤
@Generator/second/Generator/secondbatch_normalized/gamma/Adam_1:0EGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/AssignEGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/read:02RGenerator/second/Generator/secondbatch_normalized/gamma/Adam_1/Initializer/zeros:0

=Generator/second/Generator/secondbatch_normalized/beta/Adam:0BGenerator/second/Generator/secondbatch_normalized/beta/Adam/AssignBGenerator/second/Generator/secondbatch_normalized/beta/Adam/read:02OGenerator/second/Generator/secondbatch_normalized/beta/Adam/Initializer/zeros:0
 
?Generator/second/Generator/secondbatch_normalized/beta/Adam_1:0DGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/AssignDGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/read:02QGenerator/second/Generator/secondbatch_normalized/beta/Adam_1/Initializer/zeros:0

<Generator/third/Generator/thirdfully_connected/kernel/Adam:0AGenerator/third/Generator/thirdfully_connected/kernel/Adam/AssignAGenerator/third/Generator/thirdfully_connected/kernel/Adam/read:02NGenerator/third/Generator/thirdfully_connected/kernel/Adam/Initializer/zeros:0

>Generator/third/Generator/thirdfully_connected/kernel/Adam_1:0CGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/AssignCGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/read:02PGenerator/third/Generator/thirdfully_connected/kernel/Adam_1/Initializer/zeros:0

:Generator/third/Generator/thirdfully_connected/bias/Adam:0?Generator/third/Generator/thirdfully_connected/bias/Adam/Assign?Generator/third/Generator/thirdfully_connected/bias/Adam/read:02LGenerator/third/Generator/thirdfully_connected/bias/Adam/Initializer/zeros:0

<Generator/third/Generator/thirdfully_connected/bias/Adam_1:0AGenerator/third/Generator/thirdfully_connected/bias/Adam_1/AssignAGenerator/third/Generator/thirdfully_connected/bias/Adam_1/read:02NGenerator/third/Generator/thirdfully_connected/bias/Adam_1/Initializer/zeros:0

<Generator/third/Generator/thirdbatch_normalized/gamma/Adam:0AGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/AssignAGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/read:02NGenerator/third/Generator/thirdbatch_normalized/gamma/Adam/Initializer/zeros:0

>Generator/third/Generator/thirdbatch_normalized/gamma/Adam_1:0CGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/AssignCGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/read:02PGenerator/third/Generator/thirdbatch_normalized/gamma/Adam_1/Initializer/zeros:0

;Generator/third/Generator/thirdbatch_normalized/beta/Adam:0@Generator/third/Generator/thirdbatch_normalized/beta/Adam/Assign@Generator/third/Generator/thirdbatch_normalized/beta/Adam/read:02MGenerator/third/Generator/thirdbatch_normalized/beta/Adam/Initializer/zeros:0

=Generator/third/Generator/thirdbatch_normalized/beta/Adam_1:0BGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/AssignBGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/read:02OGenerator/third/Generator/thirdbatch_normalized/beta/Adam_1/Initializer/zeros:0

<Generator/forth/Generator/forthfully_connected/kernel/Adam:0AGenerator/forth/Generator/forthfully_connected/kernel/Adam/AssignAGenerator/forth/Generator/forthfully_connected/kernel/Adam/read:02NGenerator/forth/Generator/forthfully_connected/kernel/Adam/Initializer/zeros:0

>Generator/forth/Generator/forthfully_connected/kernel/Adam_1:0CGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/AssignCGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/read:02PGenerator/forth/Generator/forthfully_connected/kernel/Adam_1/Initializer/zeros:0

:Generator/forth/Generator/forthfully_connected/bias/Adam:0?Generator/forth/Generator/forthfully_connected/bias/Adam/Assign?Generator/forth/Generator/forthfully_connected/bias/Adam/read:02LGenerator/forth/Generator/forthfully_connected/bias/Adam/Initializer/zeros:0

<Generator/forth/Generator/forthfully_connected/bias/Adam_1:0AGenerator/forth/Generator/forthfully_connected/bias/Adam_1/AssignAGenerator/forth/Generator/forthfully_connected/bias/Adam_1/read:02NGenerator/forth/Generator/forthfully_connected/bias/Adam_1/Initializer/zeros:0

<Generator/forth/Generator/forthbatch_normalized/gamma/Adam:0AGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/AssignAGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/read:02NGenerator/forth/Generator/forthbatch_normalized/gamma/Adam/Initializer/zeros:0

>Generator/forth/Generator/forthbatch_normalized/gamma/Adam_1:0CGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/AssignCGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/read:02PGenerator/forth/Generator/forthbatch_normalized/gamma/Adam_1/Initializer/zeros:0

;Generator/forth/Generator/forthbatch_normalized/beta/Adam:0@Generator/forth/Generator/forthbatch_normalized/beta/Adam/Assign@Generator/forth/Generator/forthbatch_normalized/beta/Adam/read:02MGenerator/forth/Generator/forthbatch_normalized/beta/Adam/Initializer/zeros:0

=Generator/forth/Generator/forthbatch_normalized/beta/Adam_1:0BGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/AssignBGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/read:02OGenerator/forth/Generator/forthbatch_normalized/beta/Adam_1/Initializer/zeros:0

Generator/dense/kernel/Adam:0"Generator/dense/kernel/Adam/Assign"Generator/dense/kernel/Adam/read:02/Generator/dense/kernel/Adam/Initializer/zeros:0
 
Generator/dense/kernel/Adam_1:0$Generator/dense/kernel/Adam_1/Assign$Generator/dense/kernel/Adam_1/read:021Generator/dense/kernel/Adam_1/Initializer/zeros:0

Generator/dense/bias/Adam:0 Generator/dense/bias/Adam/Assign Generator/dense/bias/Adam/read:02-Generator/dense/bias/Adam/Initializer/zeros:0

Generator/dense/bias/Adam_1:0"Generator/dense/bias/Adam_1/Assign"Generator/dense/bias/Adam_1/read:02/Generator/dense/bias/Adam_1/Initializer/zeros:0"7
	summaries*
(
discriminator_loss:0
generator_loss:0"Ń*
trainable_variablesš*ś*

7Generator/first/Generator/firstfully_connected/kernel:0<Generator/first/Generator/firstfully_connected/kernel/Assign<Generator/first/Generator/firstfully_connected/kernel/read:02RGenerator/first/Generator/firstfully_connected/kernel/Initializer/random_uniform:08
ú
5Generator/first/Generator/firstfully_connected/bias:0:Generator/first/Generator/firstfully_connected/bias/Assign:Generator/first/Generator/firstfully_connected/bias/read:02GGenerator/first/Generator/firstfully_connected/bias/Initializer/zeros:08

9Generator/second/Generator/secondfully_connected/kernel:0>Generator/second/Generator/secondfully_connected/kernel/Assign>Generator/second/Generator/secondfully_connected/kernel/read:02TGenerator/second/Generator/secondfully_connected/kernel/Initializer/random_uniform:08

7Generator/second/Generator/secondfully_connected/bias:0<Generator/second/Generator/secondfully_connected/bias/Assign<Generator/second/Generator/secondfully_connected/bias/read:02IGenerator/second/Generator/secondfully_connected/bias/Initializer/zeros:08

9Generator/second/Generator/secondbatch_normalized/gamma:0>Generator/second/Generator/secondbatch_normalized/gamma/Assign>Generator/second/Generator/secondbatch_normalized/gamma/read:02JGenerator/second/Generator/secondbatch_normalized/gamma/Initializer/ones:08

8Generator/second/Generator/secondbatch_normalized/beta:0=Generator/second/Generator/secondbatch_normalized/beta/Assign=Generator/second/Generator/secondbatch_normalized/beta/read:02JGenerator/second/Generator/secondbatch_normalized/beta/Initializer/zeros:08

7Generator/third/Generator/thirdfully_connected/kernel:0<Generator/third/Generator/thirdfully_connected/kernel/Assign<Generator/third/Generator/thirdfully_connected/kernel/read:02RGenerator/third/Generator/thirdfully_connected/kernel/Initializer/random_uniform:08
ú
5Generator/third/Generator/thirdfully_connected/bias:0:Generator/third/Generator/thirdfully_connected/bias/Assign:Generator/third/Generator/thirdfully_connected/bias/read:02GGenerator/third/Generator/thirdfully_connected/bias/Initializer/zeros:08

7Generator/third/Generator/thirdbatch_normalized/gamma:0<Generator/third/Generator/thirdbatch_normalized/gamma/Assign<Generator/third/Generator/thirdbatch_normalized/gamma/read:02HGenerator/third/Generator/thirdbatch_normalized/gamma/Initializer/ones:08
ţ
6Generator/third/Generator/thirdbatch_normalized/beta:0;Generator/third/Generator/thirdbatch_normalized/beta/Assign;Generator/third/Generator/thirdbatch_normalized/beta/read:02HGenerator/third/Generator/thirdbatch_normalized/beta/Initializer/zeros:08

7Generator/forth/Generator/forthfully_connected/kernel:0<Generator/forth/Generator/forthfully_connected/kernel/Assign<Generator/forth/Generator/forthfully_connected/kernel/read:02RGenerator/forth/Generator/forthfully_connected/kernel/Initializer/random_uniform:08
ú
5Generator/forth/Generator/forthfully_connected/bias:0:Generator/forth/Generator/forthfully_connected/bias/Assign:Generator/forth/Generator/forthfully_connected/bias/read:02GGenerator/forth/Generator/forthfully_connected/bias/Initializer/zeros:08

7Generator/forth/Generator/forthbatch_normalized/gamma:0<Generator/forth/Generator/forthbatch_normalized/gamma/Assign<Generator/forth/Generator/forthbatch_normalized/gamma/read:02HGenerator/forth/Generator/forthbatch_normalized/gamma/Initializer/ones:08
ţ
6Generator/forth/Generator/forthbatch_normalized/beta:0;Generator/forth/Generator/forthbatch_normalized/beta/Assign;Generator/forth/Generator/forthbatch_normalized/beta/read:02HGenerator/forth/Generator/forthbatch_normalized/beta/Initializer/zeros:08

Generator/dense/kernel:0Generator/dense/kernel/AssignGenerator/dense/kernel/read:023Generator/dense/kernel/Initializer/random_uniform:08
~
Generator/dense/bias:0Generator/dense/bias/AssignGenerator/dense/bias/read:02(Generator/dense/bias/Initializer/zeros:08
Ť
?Discriminator/first/Discriminator/firstfully_connected/kernel:0DDiscriminator/first/Discriminator/firstfully_connected/kernel/AssignDDiscriminator/first/Discriminator/firstfully_connected/kernel/read:02ZDiscriminator/first/Discriminator/firstfully_connected/kernel/Initializer/random_uniform:08

=Discriminator/first/Discriminator/firstfully_connected/bias:0BDiscriminator/first/Discriminator/firstfully_connected/bias/AssignBDiscriminator/first/Discriminator/firstfully_connected/bias/read:02ODiscriminator/first/Discriminator/firstfully_connected/bias/Initializer/zeros:08
ł
ADiscriminator/second/Discriminator/secondfully_connected/kernel:0FDiscriminator/second/Discriminator/secondfully_connected/kernel/AssignFDiscriminator/second/Discriminator/secondfully_connected/kernel/read:02\Discriminator/second/Discriminator/secondfully_connected/kernel/Initializer/random_uniform:08
˘
?Discriminator/second/Discriminator/secondfully_connected/bias:0DDiscriminator/second/Discriminator/secondfully_connected/bias/AssignDDiscriminator/second/Discriminator/secondfully_connected/bias/read:02QDiscriminator/second/Discriminator/secondfully_connected/bias/Initializer/zeros:08

Discriminator/out/kernel:0Discriminator/out/kernel/AssignDiscriminator/out/kernel/read:025Discriminator/out/kernel/Initializer/random_uniform:08

Discriminator/out/bias:0Discriminator/out/bias/AssignDiscriminator/out/bias/read:02*Discriminator/out/bias/Initializer/zeros:08"
train_op

Adam
Adam_1CJlŽ      ŔCĹ	v #ŽýÖA*˘
Č
discriminator_loss*ą	   fü?   ŕč@      P@!   ă_9b@)^^ŰĚÁđt@2@yLúâÓ˙ů?SFiü?ÜÍ.u˙?ŕńtM@w`<f@ę6vŁď@hź5@˙˙˙˙˙˙ď:@              đ?      @      *@      9@      2@       @        
Ô
generator_loss*Á	   `ŕŔ   Ŕőż      P@!   ş9ł[Ŕ)ÁR°4Jh@2Hw`<fŔŕńtMŔÜÍ.u˙żSFiüżyLúâÓ˙ůżEĚŔ˘÷ż3?čŻ|őżşP1óż˙˙˙˙˙˙ď:H               @      @      1@      1@      .@      @      đ?        wDł