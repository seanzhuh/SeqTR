from seqtr.models import \
    (MODELS,
     build_vis_enc,
     build_lan_enc,
     build_fusion,
     build_head)
from .base import BaseModel


@MODELS.register_module()
class OneStageModel(BaseModel):
    def __init__(self,
                 word_emb,
                 num_token,
                 vis_enc,
                 lan_enc,
                 head,
                 fusion):
        super(OneStageModel, self).__init__()
        self.vis_enc = build_vis_enc(vis_enc)
        self.lan_enc = build_lan_enc(lan_enc, {'word_emb': word_emb,
                                               'num_token': num_token})
        self.head = build_head(head)
        self.fusion = build_fusion(fusion)

    def extract_visual_language(self, img, ref_expr_inds):
        y = self.lan_enc(ref_expr_inds)
        x = self.vis_enc(img, y)
        return x, y
