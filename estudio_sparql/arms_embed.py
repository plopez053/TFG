# -*- coding: utf-8 -*-
import arms as A
import embed_bank as EMB
import arms_f as F


def D_embed_dyn(pregunta):
    return A._wrap(A.SCHEMA_MIN + A.VOCAB + A.RULES_KEY, pregunta, EMB.nearest(pregunta, 3))


def F_embed(pregunta):
    return D_embed_dyn(pregunta)


ARMS_EMBED = {
    "D_embed_dyn": D_embed_dyn,
    "F_embed": F_embed,
}
POST_EMBED = {"F_embed": F.post_F}
