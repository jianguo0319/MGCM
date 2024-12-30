

@register_task("speech_to_text_multitask_MGCM")
class SpeechToTextMultitaskWithMGCM(LegacyFairseqTask):
    
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        """
        Losses is a parameter in the form of a list containing the losses of multiple tasks.
        we define the first loss in the list as the loss of the primary task and the others as the losses of the auxiliary tasks.
        """
        with torch.autograd.profiler.record_function("forward"):
            losses, sample_size, logging_output = criterion(  
                model, sample, update_num=update_num
            )


        gradient_vaccine_multi_obj(losses,optimizer)

        return sum(losses), sample_size, logging_output



def get_grads(optimizer):
    grad=[]
    has_grad=[]
    for param in optimizer.fp16_params:
        if param.grad !=None:
            grad.append(param.grad.clone())
            has_grad.append(1)
        else:
            grad.append(torch.zeros_like(param))
            has_grad.append(0)
    return grad,has_grad

def set_grads(optimizer,grad,has_grad):
    for i in range(len(has_grad)):
        optimizer.fp16_params[i].grad=grad[i]
    return

def gradient_vaccine_multi_obj(objectives,optimizer):

    assert len(objectives)  >= 2

    # get the gradient of the whole model from different objectives
    grads, has_grads = [], []
    
    for index,obj in enumerate(objectives):
        optimizer.zero_grad()
        optimizer._needs_sync = True
        if index==len(objectives)-1:
            with torch.autograd.profiler.record_function("backward"):
                obj = optimizer.scaler.scale(obj)
                obj.backward()
        else:
            with torch.autograd.profiler.record_function("backward"):
                obj = optimizer.scaler.scale(obj)
                obj.backward(retain_graph=True)
        
        grad, has_grad = get_grads(optimizer)
        grads.append(grad)
        has_grads.append(has_grad)

    # generate a deepcopy of grads
    old_grads = copy.deepcopy(grads[0])

    st_mt_sim=torch.tensor(0)
    st_asr_sim=torch.tensor(0)


    for index,(st_grad,mt_grad,asr_grad) in enumerate(zip(grads[0],grads[1],grads[2])):
        if not has_grads[0][index]:
            old_grads[index] = mt_grad + asr_grad
            continue
        original_shape = st_grad.shape
        st_grad = st_grad.view(-1)
        mt_grad = mt_grad.view(-1)
        asr_grad = asr_grad.view(-1)
        if has_grads[1][index]:
            st_mt_sim = cosine_similarity(st_grad.float(), mt_grad.float(), dim=0).half()
            if st_mt_sim < 0:
                mt_grad = mt_grad-(torch.dot(st_grad.float(),mt_grad.float())/torch.dot(st_grad.float(),st_grad.float())).half()*st_grad
        if has_grads[2][index]:
            st_asr_sim = cosine_similarity(st_grad.float(), asr_grad.float(), dim=0).half()
            if st_asr_sim < 0:
                asr_grad = asr_grad-(torch.dot(st_grad.float(),asr_grad.float())/torch.dot(st_grad.float(),st_grad.float())).half()*st_grad
        st_grad = st_grad + asr_grad + mt_grad
        st_grad = st_grad.view(original_shape)
        old_grads[index] = st_grad

    set_grads(optimizer,old_grads,has_grads[0])

    return 