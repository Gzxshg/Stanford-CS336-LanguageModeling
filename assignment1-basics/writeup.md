2.1 The Unicode Standard
(a) \x00
(b) __repr__() should enable object reconstruction
(c) the return of chr(0) can't be displayed in the printf

2.2 Unicode Encodings
(a) UTF-16 and UTF-32 separately take 16 and 32 bytes, which means they consume more storage space. Moreover, they would occur the problem about Endian.
(b) 你好啊；As UTF-8 is only play an effective role in the ASCII code
(c) \xef\xbf\xbf

3.5 Transformer Accounting
(a) Total Parameter is 1.64B. Transformer Blocks take most of them. 
(b) At the same time, the requirement of model loading is 3.52TFLOPs at single precision mode.
(c) FFN
(d) The parameter of Transformer Block grows much faster than other
(e) Total flops is sixteen times than origin.