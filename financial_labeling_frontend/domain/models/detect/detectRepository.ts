// import ApiService from '@/services/api.service'

export interface DetectRepository {
    detect(projectId: number, exampleId: number): Promise<void>;
    segment(projectId: number, exampleId: number): Promise<void>;

};