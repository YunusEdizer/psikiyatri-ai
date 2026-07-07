"""Modalite alt-modelleri: her modalite ham veriyi bir gömme vektörüne çevirir.

- ``vision`` : yüz ifadesi / mikro-ifade
- ``audio``  : ses tonu (MFCC, eGeMAPS)
- ``text``   : konuşma içeriği (Türkçe NLP)

Her modalite bir ``encode(...) -> Tensor[embed_dim]`` sözleşmesi sunar,
böylece füzyon katmanı modalitelerin iç detayından bağımsız kalır.
"""
