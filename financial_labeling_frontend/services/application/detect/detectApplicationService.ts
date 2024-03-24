import {DetectRepository } from '~/domain/models/detect/detectRepository'

export class DetectApplicationService {
    // 定义服务
    constructor(private readonly repository: DetectRepository) {}

    public async detect(project_id: number, example_id: number): Promise<void> {
      await this.repository.detect(project_id, example_id)
    }

    
    public async segment(project_id: number, example_id: number): Promise<void> {
      await this.repository.segment(project_id, example_id)
    }
}