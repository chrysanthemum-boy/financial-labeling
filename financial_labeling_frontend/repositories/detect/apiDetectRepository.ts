import ApiService from '@/services/api.service'
import { DetectRepository } from '~/domain/models/detect/detectRepository'

export class APIDetectRepository implements DetectRepository {
    // labelName = 'bboxes'
    constructor(private readonly request = ApiService) {}

  async detect(projectId:number, exampleId: number): Promise<void> {
    const url = `/projects/${projectId}/examples/${exampleId}/bboxes/detect`
    await this.request.get(url)
  }

  async segment(projectId:number, exampleId: number): Promise<void> {
    const url = `/projects/${projectId}/examples/${exampleId}/segments/detect`
    await this.request.get(url)
  }

}